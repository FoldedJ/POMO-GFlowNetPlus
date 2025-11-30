import os
import itertools
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import collections

from improver.ImproverModelParts import SharedEncoder, SharedDecoder, compute_tour_length
from improver.GFlowComponents import ValueNetwork, BacktrackPolicy, ReconstructionPolicy, ReconstructionState
from TSProblemDef import get_random_problems
from utils.utils import LogData, TimeEstimator, util_save_log_image_with_label


class TBConfig:
    """TBConfig: temperature, lr, weight_decay, train_batch_size, train_episodes, save_interval, lambda weights"""
    def __init__(self, temperature: float = 1.0, lr: float = 1e-4, weight_decay: float = 1e-6,
                 train_batch_size: int = 64, train_episodes: int = 10000, save_interval: int = 1000,
                 lambda_value: float = 0.1, lambda_tb: float = 1.0, lr_logZ: float = 1e-3,
                 k_backtrack: int = 3, replay_capacity: int = 1000, replay_fraction: float = 0.5,
                 replay_warmup: int = 100, ema_beta: float = 0.9):
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.train_episodes = train_episodes
        self.save_interval = save_interval
        self.lambda_value = lambda_value
        self.lambda_tb = lambda_tb
        self.lr_logZ = lr_logZ
        self.k_backtrack = k_backtrack
        self.replay_capacity = replay_capacity
        self.replay_fraction = replay_fraction
        self.replay_warmup = replay_warmup
        self.ema_beta = ema_beta


class GFlowTBTrainer(nn.Module):
    """Trajectory Balance 训练器：对一次回溯-重构轨迹进行 TB 损失训练"""
    def __init__(self, model_params: Dict[str, Any], tb_cfg: TBConfig):
        super().__init__()
        self.model_params = model_params
        self.tb_cfg = tb_cfg

        self.encoder = SharedEncoder(**model_params)  # 共享编码器：坐标->嵌入
        self.decoder = SharedDecoder(**model_params)  # 共享解码器：生成下一城市分布
        self.value_net = ValueNetwork(self.encoder)   # 价值网络（可用于评估）
        self.backtrack = BacktrackPolicy(self.encoder)  # 回溯策略：位置打分
        self.reconstruct = ReconstructionPolicy(self.decoder)  # 重构策略：基于前缀生成后缀

        self.logZ = nn.Parameter(torch.zeros(1))  # TB 的配分常数对数参数

        # 仅加入一次性参数，避免重复注册导致优化器警告
        unique_params = []
        seen = set()
        for p in itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.value_net.head.parameters(),  # 仅价值头参数
                self.backtrack.head.parameters(),  # 仅回溯头参数
        ):
            if id(p) not in seen:
                unique_params.append(p)
                seen.add(id(p))
        self.optimizer = torch.optim.Adam([
            {'params': unique_params, 'lr': tb_cfg.lr, 'weight_decay': tb_cfg.weight_decay},
            {'params': [self.logZ], 'lr': tb_cfg.lr_logZ, 'weight_decay': 0.0},
        ])

        self.result_log = LogData()  # 日志记录器
        self.time_estimator = TimeEstimator()  # 训练时间估计
        self.replay = collections.deque(maxlen=self.tb_cfg.replay_capacity)
        self.ema_logR = None

    def build_initial_tour(self, problems: torch.Tensor) -> torch.Tensor:
        """Input: (batch, problem, 2) -> Output: (batch, problem) 初始路径为随机排列
        说明：按样本生成 0..N-1 的随机排列，作为提升式算法的基础路径
        """
        device = problems.device
        batch = problems.size(0)
        N = problems.size(1)
        tours = torch.stack([torch.randperm(N, device=device) for _ in range(batch)], dim=0)
        return tours

    def _sample_backtrack_point(self, encoded_nodes: torch.Tensor, tour: torch.Tensor, temperature: float):
        """Return (indices, logprob) of sampled backtrack point under temperature-softmax"""
        logits = self.backtrack.forward_with_encoded(encoded_nodes, tour)  # 回溯位置logits（使用已编码节点）
        log_probs = F.log_softmax(logits / max(temperature, 1e-6), dim=1)  # 温度软化后的log概率
        idx = torch.distributions.Categorical(logits=log_probs).sample()  # 抽样一个回溯点
        return idx, log_probs[torch.arange(log_probs.size(0)), idx]

    def _sample_backtrack_points(self, encoded_nodes: torch.Tensor, tour: torch.Tensor, temperature: float, k: int):
        logits = self.backtrack.forward_with_encoded(encoded_nodes, tour)
        log_probs = F.log_softmax(logits / max(temperature, 1e-6), dim=1)
        samples = torch.distributions.Categorical(logits=log_probs).sample((k,))
        idxs = samples.transpose(0, 1).contiguous()
        b_idx = torch.arange(log_probs.size(0))[:, None].expand(log_probs.size(0), k)
        lp = log_probs[b_idx, idxs]
        return idxs, lp

    def _sample_reconstruction(self, problems: torch.Tensor, encoded_nodes: torch.Tensor,
                               initial: torch.Tensor, prefix_len: torch.Tensor):
        """按批逐个重构后缀（保持前缀顺序），返回 (tour, logprob_sum)
        - 解决不同样本前缀长度不一致导致的 stack 问题
        """
        batch, problem_size, _ = problems.shape
        tours = []
        logprobs = []
        for b in range(batch):
            pref = initial[b, :prefix_len[b]]  # 取该样本前缀
            state = ReconstructionState.from_prefix(encoded_nodes[b:b+1, :, :], pref.unsqueeze(0))
            tour_b, logprob_b = self.reconstruct.reconstruct_suffix_state(
                problems[b:b+1, :, :], state, prefix=pref.unsqueeze(0), eval_type='softmax', return_logprob=True
            )
            tours.append(tour_b.squeeze(0))
            logprobs.append(logprob_b.squeeze(0))
        tour = torch.stack(tours, dim=0)
        logprob_sum = torch.stack(logprobs, dim=0)
        return tour, logprob_sum

    def tb_loss_for_batch(self, problems: torch.Tensor) -> Dict[str, Any]:
        """Compute TB loss for one batch and report metrics: loss, avg_len, avg_logP, avg_logR"""
        batch = problems.size(0)
        initial = self.build_initial_tour(problems)  # 基线初始路径
        encoded_nodes = self.encoder(problems)  # 图编码

        k = self.tb_cfg.k_backtrack
        bt_idxs, bt_logprobs = self._sample_backtrack_points(encoded_nodes, initial, self.tb_cfg.temperature, k)
        tb_list = []
        val_list = []
        avg_len_list = []
        avg_logP_list = []
        avg_logR_list = []
        for t in range(k):
            prefix_len = bt_idxs[:, t] + 1
            recon_tour, recon_logprob = self._sample_reconstruction(problems, encoded_nodes, initial, prefix_len)
            lengths = compute_tour_length(problems, recon_tour)
            reward = torch.exp(-lengths)
            log_R = torch.log(reward + 1e-12)
            log_P = bt_logprobs[:, t] + recon_logprob
            tb = (self.logZ + log_P - log_R) ** 2

            device = problems.device
            visited_mask = torch.zeros(batch, problems.size(1), dtype=torch.bool, device=device)
            last_idx = torch.zeros(batch, dtype=torch.long, device=device)
            for b in range(batch):
                pl = prefix_len[b].item()
                pref = initial[b, :pl]
                visited_mask[b, pref] = True
                last_idx[b] = pref[-1]
            pred = self.value_net(problems, visited_mask, last_idx)
            v_loss = F.mse_loss(pred, lengths.detach())

            tb_list.append(tb)
            val_list.append(v_loss)
            avg_len_list.append(lengths.mean().item())
            avg_logP_list.append(log_P.mean().item())
            avg_logR_list.append(log_R.mean().item())

        loss_tb = torch.stack(tb_list, dim=1).mean()
        value_loss = torch.stack(val_list).mean()
        loss = self.tb_cfg.lambda_tb * loss_tb + self.tb_cfg.lambda_value * value_loss
        avg_logR = sum(avg_logR_list) / len(avg_logR_list)
        if self.ema_logR is None:
            self.ema_logR = avg_logR
        else:
            self.ema_logR = self.tb_cfg.ema_beta * self.ema_logR + (1 - self.tb_cfg.ema_beta) * avg_logR
        avg_adv = avg_logR - self.ema_logR
        return {
            'loss': loss,
            'tb_loss': loss_tb.item(),
            'value_loss': value_loss.item(),
            'avg_len': sum(avg_len_list) / len(avg_len_list),
            'avg_logP': sum(avg_logP_list) / len(avg_logP_list),
            'avg_logR': sum(avg_logR_list) / len(avg_logR_list),
            'avg_adv': avg_adv,
        }

    def run(self, save_dir: str, epochs: int, img_style_file: str = 'style_loss_1.json'):
        """Run TB training loop with logging and checkpoint saving"""
        os.makedirs(save_dir, exist_ok=True)
        self.time_estimator.reset(epochs)
        for ep in range(1, epochs + 1):
            new_problems = get_random_problems(self.tb_cfg.train_batch_size, self.model_params['problem_size'])
            use_replay = len(self.replay) >= self.tb_cfg.replay_warmup
            if use_replay:
                r = int(self.tb_cfg.replay_fraction * self.tb_cfg.train_batch_size)
                r = min(r, len(self.replay))
                idx = torch.randperm(len(self.replay))[:r]
                rep_list = [self.replay[i] for i in idx.tolist()]
                rep_stack = torch.stack(rep_list, dim=0).to(new_problems.device)
                problems = torch.cat([new_problems, rep_stack], dim=0)
            else:
                problems = new_problems
            self.train()
            self.optimizer.zero_grad()
            out = self.tb_loss_for_batch(problems)
            out['loss'].backward()
            self.optimizer.step()

            self.result_log.append('train_loss', out['loss'].item())  # 记录训练指标
            self.result_log.append('train_avg_len', out['avg_len'])
            self.result_log.append('train_avg_logP', out['avg_logP'])
            self.result_log.append('train_avg_logR', out['avg_logR'])
            self.result_log.append('train_tb_loss', out['tb_loss'])
            self.result_log.append('train_value_loss', out['value_loss'])
            self.result_log.append('train_avg_adv', out.get('avg_adv', 0.0))

            if ep % self.tb_cfg.save_interval == 0:
                ckpt = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'value_net': self.value_net.state_dict(),
                    'backtrack': self.backtrack.state_dict(),
                    'reconstruct': self.reconstruct.state_dict(),
                    'logZ': self.logZ.detach().cpu(),
                    'result_log': self.result_log.get_raw_data(),
                }
                torch.save(ckpt, os.path.join(save_dir, f'checkpoint-{ep}.pt'))  # 保存checkpoint

            self.time_estimator.print_est_time(ep, epochs)
            to_store = new_problems.detach().cpu()
            for i in range(to_store.size(0)):
                self.replay.append(to_store[i])

            # 追加详细训练日志到 run_log
            logger = logging.getLogger('root')
            logger.info(
                "Epoch {}/{}: loss={:.6f}, tb_loss={:.6f}, value_loss={:.6f}, avg_len={:.6f}, avg_logP={:.6f}, avg_logR={:.6f}".format(
                    ep, epochs,
                    out['loss'].item() if hasattr(out['loss'], 'item') else float(out['loss']),
                    out.get('tb_loss', 0.0),
                    out.get('value_loss', 0.0),
                    out['avg_len'], out['avg_logP'], out['avg_logR']
                )
            )

        util_save_log_image_with_label(
            result_file_prefix=os.path.join(save_dir, 'train'),
            img_params={'json_foldername': 'log_image_style', 'filename': img_style_file},
            result_log=self.result_log,
            labels=['train_loss', 'train_tb_loss', 'train_value_loss', 'train_avg_len']  # 输出训练曲线
        )

