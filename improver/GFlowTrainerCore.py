import os
import itertools
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from improver.ImproverModelParts import SharedEncoder, compute_tour_length, compute_remaining_length
from improver.GFlowComponents import ValueNetwork, sample_backtrack_points, sample_reconstruction_by_edge_split
from TSProblemDef import get_random_problems
from utils.utils import LogData, TimeEstimator, util_save_log_image_with_label, AverageMeter


class TBConfig:

    def __init__(self, temperature: float = 1.0, lr: float = 1e-4, weight_decay: float = 1e-6,
                 train_batch_size: int = 64, save_interval: int = 1000,
                 lambda_value: float = 0.1, lambda_tb: float = 1.0, lr_logZ: float = 1e-3,
                 k_backtrack: int = 3, m_reconstruct: int = 3, episode_steps: int = 1,
                 train_episodes: int = 1000):
        self.temperature = temperature  # 温度参数
        self.lr = lr  # 学习率
        self.weight_decay = weight_decay  # 权重衰减（正则化）
        self.train_batch_size = train_batch_size  # 训练批次大小
        self.save_interval = save_interval  # 保存间隔
        self.lambda_value = lambda_value  # 价值网络损失权重
        self.lambda_tb = lambda_tb  # TB 损失权重
        self.lr_logZ = lr_logZ  # 配分常数学习率
        self.k_backtrack = k_backtrack  # 回溯时考虑的候选数
        self.m_reconstruct = m_reconstruct  # 重构时考虑的候选数
        self.episode_steps = episode_steps  # 每个episode的优化步数
        self.train_episodes = train_episodes # 每个epoch的episode数


class GFlowTBTrainer(nn.Module):
    
    def __init__(self, model_params: Dict[str, Any], tb_cfg: TBConfig):
        super().__init__()
        self.model_params = model_params
        self.tb_cfg = tb_cfg

        self.encoder = SharedEncoder(**model_params)
        self.value_net = ValueNetwork(self.encoder)

        self.logZ = nn.Parameter(torch.zeros(1))  # TB 的配分常数对数参数

        # 仅加入一次性参数，避免重复注册导致优化器警告
        unique_params = []
        seen = set()
        for p in itertools.chain(
                self.encoder.parameters(),
                self.value_net.head.parameters(),
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

    def build_initial_tour(self, problems: torch.Tensor) -> torch.Tensor:
        device = problems.device
        batch = problems.size(0)
        N = problems.size(1)
        tours = torch.stack([torch.randperm(N, device=device) for _ in range(batch)], dim=0)
        return tours
    

    def tb_loss_for_batch(self, problems: torch.Tensor, initial: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        计算 TB 损失
        """
        self.train()
        self.optimizer.zero_grad()
        batch = problems.size(0)
        if initial is None:
            initial = self.build_initial_tour(problems)
        
        # 读取参数
        k = self.tb_cfg.k_backtrack
        m = self.tb_cfg.m_reconstruct
        steps = self.tb_cfg.episode_steps
        
        tb_list = [] # TB 损失
        val_list = [] # 价值网络的MSE损失    
        avg_len_list = [] # 平均路径长度
        avg_logP_list = [] # pf - pb
        avg_logR_list = [] # log_R
        adv_list = [] # 优势

        for _ in range(steps):
            # 抽样回溯点
            bt_idxs, bt_logprob_mat = sample_backtrack_points(problems, initial, self.value_net, k, self.tb_cfg.temperature)
            bt_hist = torch.bincount(bt_idxs.reshape(-1), minlength=problems.size(1)).detach().cpu().tolist()
            
            candidates = [] # 重构候选路径
            cand_lengths = [] # 候选路径长度
            # 遍历每个回溯点，生成重构候选路径
            for t in range(k):
                # 前缀长度（已访问城市数）
                prefix_len = bt_idxs[:, t] + 1
                device = problems.device
                # 构造价值网络输入
                visited_mask = torch.zeros(batch, problems.size(1), dtype=torch.bool, device=device)
                last_idx = torch.zeros(batch, dtype=torch.long, device=device)      
                for b in range(batch):
                    pl = prefix_len[b].item()
                    pref = initial[b, :pl]
                    visited_mask[b, pref] = True
                    last_idx[b] = pref[-1]
                pred = self.value_net(problems, tour=initial, mask=visited_mask)
                # 按照“拆边重插”进行重构候选采样
                cand_tensor, recon_logprob_mat, recon_edge_idx_mat = sample_reconstruction_by_edge_split(
                    problems, initial, bt_idxs[:, t], m, self.value_net, self.tb_cfg.temperature
                )
                # 直方统计：回溯城市与拆分边分布
                bt_city = initial[torch.arange(batch), bt_idxs[:, t]]
                city_hist = torch.bincount(bt_city, minlength=problems.size(1)).detach().cpu().tolist()
                edges_flat = recon_edge_idx_mat.reshape(-1).clamp(min=0)
                edge_hist = torch.bincount(edges_flat, minlength=problems.size(1)).detach().cpu().tolist()
                
                # 针对每个候选路径计算损失
                for j in range(m):
                    recon_tour = cand_tensor[:, j, :]
                    lengths = compute_tour_length(problems, recon_tour)
                    log_R = -lengths
                    log_pb = bt_logprob_mat[torch.arange(batch), bt_idxs[:, t]]
                    log_pf = recon_logprob_mat[:, j]
                    # 计算 TB 损失
                    tb = (self.logZ + log_pf - log_R - log_pb) ** 2
                    # 价值网络MSE损失
                    remaining = compute_remaining_length(problems, recon_tour, prefix_len)
                    v_loss = F.mse_loss(pred, remaining.detach())
                    
                    tb_list.append(tb)
                    val_list.append(v_loss)
                    avg_len_list.append(lengths.mean().item())
                    avg_logP_list.append((log_pf - log_pb).mean().item())
                    avg_logR_list.append(log_R.mean().item())
                    candidates.append(recon_tour)
                    cand_lengths.append(lengths)
            # 计算优势并更新初始路径
            if candidates:
                C = len(candidates) # 候选路径总数（k×m）
                cand_stack = torch.stack(candidates, dim=0) # (C, batch, N) → 所有候选路径
                len_stack = torch.stack(cand_lengths, dim=1) # (batch, C) → 每个样本的所有候选长度
                b_shared = len_stack.mean(dim=1) # (batch,) → 共享基线
                # 优势
                A = b_shared.unsqueeze(1) - len_stack
                adv_list.append(A.mean().item())
                # 选最优并更新初始路径
                best_idx = torch.argmax(A, dim=1)
                initial = cand_stack[best_idx, torch.arange(batch), :]

        # 汇总损失
        loss_tb = torch.stack(tb_list, dim=1).mean() if tb_list else torch.tensor(0.0, device=problems.device)
        value_loss = torch.stack(val_list).mean() if val_list else torch.tensor(0.0, device=problems.device)
        loss = self.tb_cfg.lambda_tb * loss_tb + self.tb_cfg.lambda_value * value_loss
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss,
            'tb_loss': loss_tb.item(),
            'value_loss': value_loss.item(),
            'avg_len': sum(avg_len_list) / len(avg_len_list),
            'avg_logP': sum(avg_logP_list) / len(avg_logP_list),
            'avg_logR': sum(avg_logR_list) / len(avg_logR_list),
            'avg_adv': (sum(adv_list) / len(adv_list)) if adv_list else 0.0,
            'bt_points_hist': bt_hist, # 回溯点分布
            'recon_city_hist': city_hist, # 重构城市分布
            'recon_edge_hist': edge_hist, # 重构边分布
            'initial': initial,
        }


    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        loss_am = AverageMeter()
        tb_am = AverageMeter()
        val_am = AverageMeter()
        len_am = AverageMeter()
        logp_am = AverageMeter()
        logr_am = AverageMeter()
        adv_am = AverageMeter()
        problem_size = self.model_params['problem_size']
        bt_hist_accum = torch.zeros(problem_size, dtype=torch.long)
        rc_city_hist_accum = torch.zeros(problem_size, dtype=torch.long)
        rc_edge_hist_accum = torch.zeros(problem_size, dtype=torch.long)
        train_num_episode = self.tb_cfg.train_episodes
        problems = get_random_problems(self.tb_cfg.train_batch_size, self.model_params['problem_size'], seed=None)
        initial = self.build_initial_tour(problems)
        for _ in range(train_num_episode):
            out = self.tb_loss_for_batch(problems, initial=initial)
            batch_size = problems.size(0)
            loss_am.update(out['loss'].item(), batch_size)
            tb_am.update(out['tb_loss'], batch_size)
            val_am.update(out['value_loss'], batch_size)
            len_am.update(out['avg_len'], batch_size)
            logp_am.update(out['avg_logP'], batch_size)
            logr_am.update(out['avg_logR'], batch_size)
            adv_am.update(out.get('avg_adv', 0.0), batch_size)
            if 'bt_points_hist' in out and isinstance(out['bt_points_hist'], list):
                bt_hist_accum += torch.tensor(out['bt_points_hist'][:problem_size])
            if 'recon_city_hist' in out and isinstance(out['recon_city_hist'], list):
                rc_city_hist_accum += torch.tensor(out['recon_city_hist'][:problem_size])
            if 'recon_edge_hist' in out and isinstance(out['recon_edge_hist'], list):
                rc_edge_hist_accum += torch.tensor(out['recon_edge_hist'][:problem_size])
            initial = out.get('initial', initial)
        return {
            'loss': loss_am.avg,
            'tb_loss': tb_am.avg,
            'value_loss': val_am.avg,
            'avg_len': len_am.avg,
            'avg_logP': logp_am.avg,
            'avg_logR': logr_am.avg,
            'avg_adv': adv_am.avg,
            'bt_points_hist': bt_hist_accum.tolist(),
            'recon_city_hist': rc_city_hist_accum.tolist(),
            'recon_edge_hist': rc_edge_hist_accum.tolist(),
        }

    def run(self, save_dir: str, epochs: int, img_style_file: str = 'style_loss_1.json'):

        os.makedirs(save_dir, exist_ok=True)
        self.time_estimator.reset(epochs)
        for ep in range(1, epochs + 1):
            stats = self._train_one_epoch(ep)
            self.result_log.append('train_loss', stats['loss'])
            self.result_log.append('train_avg_len', stats['avg_len'])
            self.result_log.append('train_avg_logP', stats['avg_logP'])
            self.result_log.append('train_avg_logR', stats['avg_logR'])
            self.result_log.append('train_tb_loss', stats['tb_loss'])
            self.result_log.append('train_value_loss', stats['value_loss'])
            self.result_log.append('train_avg_adv', stats.get('avg_adv', 0.0))

            if ep % self.tb_cfg.save_interval == 0:
                ckpt = {
                    'encoder': self.encoder.state_dict(),
                    'value_net': self.value_net.state_dict(),
                    'logZ': self.logZ.detach().cpu(),
                    'result_log': self.result_log.get_raw_data(),
                }
                torch.save(ckpt, os.path.join(save_dir, f'checkpoint-{ep}.pt'))

            self.time_estimator.print_est_time(ep, epochs)
            logger = logging.getLogger('root')
            logger.info(
                "Epoch {}/{}: loss={:.6f}, tb_loss={:.6f}, value_loss={:.6f}, avg_len={:.6f}, avg_logP={:.6f}, avg_logR={:.6f}, bt_hist={}, rc_city_hist={}, rc_edge_hist={}".format(
                    ep, epochs,
                    stats['loss'],
                    stats['tb_loss'],
                    stats['value_loss'],
                    stats['avg_len'], stats['avg_logP'], stats['avg_logR'],
                    stats.get('bt_points_hist', []),
                    stats.get('recon_city_hist', []), stats.get('recon_edge_hist', [])
                )
            )

        util_save_log_image_with_label(
            result_file_prefix=os.path.join(save_dir, 'train'),
            img_params={'json_foldername': 'log_image_style', 'filename': img_style_file},
            result_log=self.result_log,
            labels=['train_loss', 'train_tb_loss', 'train_value_loss', 'train_avg_len']
        )
