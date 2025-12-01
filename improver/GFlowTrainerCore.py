import os
import itertools
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import collections

from improver.ImproverModelParts import SharedEncoder, SharedDecoder, compute_tour_length, compute_remaining_length, compute_suffix_lengths
from improver.GFlowComponents import ValueNetwork, BacktrackPolicy, ReconstructionPolicy, OptimizationState, sample_topk_from_logits
from TSProblemDef import get_random_problems
from utils.utils import LogData, TimeEstimator, util_save_log_image_with_label


class TBConfig:

    def __init__(self, temperature: float = 1.0, lr: float = 1e-4, weight_decay: float = 1e-6,
                 train_batch_size: int = 64, train_episodes: int = 10000, save_interval: int = 1000,
                 lambda_value: float = 0.1, lambda_tb: float = 1.0, lr_logZ: float = 1e-3,
                 lambda_consistency: float = 0.1,
                 k_backtrack: int = 3, m_reconstruct: int = 3, episode_steps: int = 1,
                 replay_capacity: int = 1000, replay_fraction: float = 0.5,
                 replay_warmup: int = 100, ema_beta: float = 0.9):
        self.temperature = temperature  # 温度参数
        self.lr = lr  # 学习率
        self.weight_decay = weight_decay  # 权重衰减（正则化）
        self.train_batch_size = train_batch_size  # 训练批次大小
        self.train_episodes = train_episodes  # 训练总episode数
        self.save_interval = save_interval  # 保存间隔
        self.lambda_value = lambda_value  # 价值网络损失权重
        self.lambda_tb = lambda_tb  # TB 损失权重
        self.lr_logZ = lr_logZ  # 配分常数学习率
        self.lambda_consistency = lambda_consistency  # 一致性损失权重
        self.k_backtrack = k_backtrack  # 回溯时考虑的候选数
        self.m_reconstruct = m_reconstruct  # 重构时考虑的候选数
        self.episode_steps = episode_steps  # 每个episode的最大步数
        self.replay_capacity = replay_capacity  # 回放缓冲区容量
        self.replay_fraction = replay_fraction  # 回放样本比例
        self.replay_warmup = replay_warmup  # 回放预热步数
        self.ema_beta = ema_beta  # EMA 平滑参数


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
        按样本生成 0..N-1 的随机排列，作为提升式算法的基础路径
        """
        device = problems.device
        batch = problems.size(0)
        N = problems.size(1)
        tours = torch.stack([torch.randperm(N, device=device) for _ in range(batch)], dim=0)
        return tours
    def _sample_backtrack_points(self, problems: torch.Tensor, tour: torch.Tensor, temperature: float, k: int):
        """根据当前路径和编码节点，按温度-softmax抽样k个回溯点
        输入：
            problems: (batch, problem, 2) 问题实例（坐标）
            tour: (batch, problem) 当前路径
            temperature: 温度参数（控制探索与利用）
            k: 抽样数量
        返回：
            idxs: (batch, k) 抽样的回溯点索引
            log_probs: (batch, k) 对应回溯点的log概率
        """
        batch, problem_size, _ = problems.shape
        # 计算每个位置的“后缀实际长度”矩阵：(batch, problem)
        suffix_len = compute_suffix_lengths(problems, tour)
        # 构造每个位置的前缀掩码与末节点索引
        visited_masks = torch.zeros(batch, problem_size, problem_size, dtype=torch.bool, device=problems.device)
        last_idx_flat = torch.zeros(batch * problem_size, dtype=torch.long, device=problems.device)
        for b in range(batch):
            seq = tour[b]
            for t in range(problem_size):
                pref = seq[:t+1]
                visited_masks[b, t, pref] = True
                last_idx_flat[b * problem_size + t] = pref[-1]
        # 并行调用价值网络
        problems_flat = problems.unsqueeze(1).expand(batch, problem_size, problem_size, 2).reshape(batch * problem_size, problem_size, 2)
        visited_flat = visited_masks.reshape(batch * problem_size, problem_size)
        # 预测每个前缀状态的“剩余长度”，再还原到 (batch, problem)
        pred_remaining = self.value_net(problems_flat, visited_flat, last_idx_flat).reshape(batch, problem_size)
        # 价值引导回溯潜力：真实后缀长度 - 预测剩余长度
        phi = suffix_len - pred_remaining
        logits = phi
        # 温度软化后抽样不重复的 top-k 索引
        idxs = sample_topk_from_logits(logits, k=k, temperature=temperature)
        # 提取所选索引的 log 概率，供损失计算使用
        log_probs = F.log_softmax(logits / max(temperature, 1e-6), dim=1)
        b_idx = torch.arange(log_probs.size(0))[:, None].expand(log_probs.size(0), k)
        lp = log_probs[b_idx, idxs]
        return idxs, lp

    def _sample_reconstruction(self, problems: torch.Tensor, encoded_nodes: torch.Tensor,
                               initial: torch.Tensor, prefix_len: torch.Tensor):
        """根据当前路径和编码节点，按温度-softmax抽样k个回溯点
        输入：
            problems: (batch, problem, 2) 问题实例（坐标）
            encoded_nodes: (batch, problem, emb_dim) 编码后的节点表示
            initial: (batch, problem) 初始路径
            prefix_len: (batch) 每个样本的前缀长度
        返回：
            tour: (batch, problem) 重构的路径
            logprob_sum: (batch) 对应路径的log概率总和
        """
        batch, problem_size, _ = problems.shape
        tours = []
        logprobs = []
        for b in range(batch):
            pref = initial[b, :prefix_len[b]]
            opt_state_b = OptimizationState(problems[b:b+1, :, :], initial[b:b+1, :], encoded_nodes[b:b+1, :, :])
            tour_b, logprob_b = self.reconstruct.reconstruct_suffix_state(
                problems[b:b+1, :, :], opt_state_b, prefix=pref.unsqueeze(0), eval_type='softmax', return_logprob=True
            )
            tours.append(tour_b.squeeze(0))
            logprobs.append(logprob_b.squeeze(0))
        tour = torch.stack(tours, dim=0)
        logprob_sum = torch.stack(logprobs, dim=0)
        return tour, logprob_sum

    def tb_loss_for_batch(self, problems: torch.Tensor) -> Dict[str, Any]:
        batch = problems.size(0)
        initial = self.build_initial_tour(problems)
        encoded_nodes = self.encoder(problems)

        k = self.tb_cfg.k_backtrack
        m = self.tb_cfg.m_reconstruct
        steps = self.tb_cfg.episode_steps
        tb_list = []
        val_list = []
        cons_list = []
        avg_len_list = []
        avg_logP_list = []
        avg_logR_list = []

        for _ in range(steps):
            bt_idxs, bt_logprobs = self._sample_backtrack_points(problems, initial, self.tb_cfg.temperature, k)
            candidates = []
            cand_lengths = []
            for t in range(k):
                prefix_len = bt_idxs[:, t] + 1
                device = problems.device
                visited_mask = torch.zeros(batch, problems.size(1), dtype=torch.bool, device=device)
                last_idx = torch.zeros(batch, dtype=torch.long, device=device)
                for b in range(batch):
                    pl = prefix_len[b].item()
                    pref = initial[b, :pl]
                    visited_mask[b, pref] = True
                    last_idx[b] = pref[-1]
                pred = self.value_net(problems, visited_mask, last_idx)
                for _ in range(m):
                    recon_tour, recon_logprob = self._sample_reconstruction(problems, encoded_nodes, initial, prefix_len)
                    lengths = compute_tour_length(problems, recon_tour)
                    reward = torch.exp(-lengths)
                    log_R = torch.log(reward + 1e-12)
                    # 轨迹平衡公式：logZ + sum log P_F - log R - sum log P_B
                    log_pb = bt_logprobs[:, t]        # 回溯选择（后向）log概率
                    log_pf = recon_logprob            # 重构后缀（前向）log概率
                    tb = (self.logZ + log_pf - log_R - log_pb) ** 2
                    remaining = compute_remaining_length(problems, recon_tour, prefix_len)
                    v_loss = F.mse_loss(pred, remaining.detach())
                    def _norm(x):
                        x_ = x - x.mean()
                        return x_ / (x_.std() + 1e-6)
                    # 策略-价值一致性：前向概率与价值方向对齐
                    cons = (_norm(log_pf) - _norm(-pred)).pow(2).mean()
                    tb_list.append(tb)
                    val_list.append(v_loss)
                    cons_list.append(cons)
                    avg_len_list.append(lengths.mean().item())
                    # 记录组合指标：平均(logPF - logPB)，用于观察与TB一致的前向-后向差
                    avg_logP_list.append((log_pf - log_pb).mean().item())
                    avg_logR_list.append(log_R.mean().item())
                    candidates.append(recon_tour)
                    cand_lengths.append(lengths)
            if candidates:
                C = len(candidates)
                cand_stack = torch.stack(candidates, dim=0)
                len_stack = torch.stack(cand_lengths, dim=1)
                best_idx = torch.argmin(len_stack, dim=1)
                initial = cand_stack[best_idx, torch.arange(batch), :]
                encoded_nodes = self.encoder(problems)

        # 汇总三类损失：TB损失、价值MSE、一致性损失，并按权重合成总损失
        loss_tb = torch.stack(tb_list, dim=1).mean() if tb_list else torch.tensor(0.0, device=problems.device)
        value_loss = torch.stack(val_list).mean() if val_list else torch.tensor(0.0, device=problems.device)
        consistency_loss = torch.stack(cons_list).mean() if cons_list else torch.tensor(0.0, device=problems.device)
        loss = self.tb_cfg.lambda_tb * loss_tb + self.tb_cfg.lambda_value * value_loss + self.tb_cfg.lambda_consistency * consistency_loss
        # 奖励的批次均值与EMA平滑，用于计算优势
        avg_logR = sum(avg_logR_list) / len(avg_logR_list)
        if self.ema_logR is None:
            self.ema_logR = avg_logR
        else:
            self.ema_logR = self.tb_cfg.ema_beta * self.ema_logR + (1 - self.tb_cfg.ema_beta) * avg_logR
        avg_adv = avg_logR - self.ema_logR
        # 返回训练指标：总损失与分项损失、平均长度/概率/奖励、优势
        return {
            'loss': loss,
            'tb_loss': loss_tb.item(),            # 轨迹平衡损失（均值）
            'value_loss': value_loss.item(),      # 价值网络损失（均值）
            'consistency_loss': consistency_loss.item(),  # 策略-价值一致性损失（均值）
            'avg_len': sum(avg_len_list) / len(avg_len_list),      # 平均路径长度
            'avg_logP': sum(avg_logP_list) / len(avg_logP_list),   # 平均轨迹log概率
            'avg_logR': sum(avg_logR_list) / len(avg_logR_list),   # 平均log奖励
            'avg_adv': avg_adv,                                    # 奖励优势（相对EMA基线）
        }

    def run(self, save_dir: str, epochs: int, img_style_file: str = 'style_loss_1.json'):
        """Run TB training loop with logging and checkpoint saving"""
        os.makedirs(save_dir, exist_ok=True)
        self.time_estimator.reset(epochs)
        # 主训练循环：采样数据、混入回放、前向-反向-更新、记录与保存
        for ep in range(1, epochs + 1):
            new_problems = get_random_problems(self.tb_cfg.train_batch_size, self.model_params['problem_size'])
            # 判断是否使用经验回放（达到预热阈值后混入历史样本）
            use_replay = len(self.replay) >= self.tb_cfg.replay_warmup
            if use_replay:
                # 计算回放采样数量并随机抽取，拼接到当前批次
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
            self.result_log.append('train_consistency_loss', out.get('consistency_loss', 0.0))
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

            # 估算剩余时间并输出；将本轮新样本加入回放队列
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

        # 训练结束后输出指标曲线图像
        util_save_log_image_with_label(
            result_file_prefix=os.path.join(save_dir, 'train'),
            img_params={'json_foldername': 'log_image_style', 'filename': img_style_file},
            result_log=self.result_log,
            labels=['train_loss', 'train_tb_loss', 'train_value_loss', 'train_consistency_loss', 'train_avg_len']  # 输出训练曲线
        )

