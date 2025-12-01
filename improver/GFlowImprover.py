import torch
import torch.nn as nn
from typing import Dict, Any

from improver.ImproverModelParts import SharedEncoder, SharedDecoder, compute_tour_length, compute_suffix_lengths
from improver.GFlowComponents import ValueNetwork, BacktrackPolicy, ReconstructionPolicy, sample_topk_from_logits, OptimizationState
from TSProblemDef import augment_xy_data_by_8_fold


class GFlowImprover:
    """提升式优化器：一次迭代执行共同回溯与并行重构，返回优中选优的路径。
    """
    def __init__(self, model_params: Dict[str, Any], k_backtrack: int = 3, m_reconstruct: int = 3, temperature: float = 1.0, use_augmentation: bool = False):
        self.encoder = SharedEncoder(**model_params)  # 共享编码器：坐标->嵌入
        self.decoder = SharedDecoder(**model_params)  # 共享解码器：生成下一城市分布
        self.value_net = ValueNetwork(self.encoder)   # 价值网络（可用于评估）
        self.backtrack = BacktrackPolicy(self.encoder)  # 回溯策略：位置打分
        self.reconstruct = ReconstructionPolicy(self.decoder)  # 重构策略：基于前缀生成后缀

        self.k_backtrack = k_backtrack  # 回溯时考虑的候选数
        self.m_reconstruct = m_reconstruct  # 重构时考虑的候选数
        self.temperature = temperature  # 温度参数（控制探索与利用）
        self.use_augmentation = use_augmentation  # 是否使用数据增强

    @torch.no_grad()
    def improve_once(self, problems: torch.Tensor, initial_tour: torch.Tensor) -> Dict[str, Any]:
        """单次提升迭代：共同回溯与并行重构，返回优中选优路径。
        problems: (batch, problem, 2) 坐标输入
        initial_tour: (batch, problem) 初始路径（可选，默认随机）
        返回: dict 包含 best_tours, best_lengths, initial_lengths, improved
        best_tours: (batch, problem) 最优路径
        best_lengths: (batch,) 最优路径长度
        initial_lengths: (batch,) 初始路径长度
        improved: (batch,) 是否有改进（初始路径长度 > 最优路径长度）
        """
        device = problems.device
        if problems.dim() != 3 or problems.size(-1) != 2:
            raise ValueError("problems must be (batch, problem, 2)")
        if initial_tour.dim() != 2:
            raise ValueError("initial_tour must be (batch, problem)")
        batch, problem_size, _ = problems.shape
        if initial_tour.size(0) != batch or initial_tour.size(1) != problem_size:
            raise ValueError("initial_tour must match problems batch and problem_size")

        if not self.use_augmentation:
            # 1) 计算每个位置的“后缀实际长度” L_actual(π[t:N])
            suffix_len = compute_suffix_lengths(problems, initial_tour)
            # 2) 构造每个前缀的访问掩码与末节点索引，用于价值网络评估剩余长度
            visited_masks = torch.zeros(batch, problem_size, problem_size, dtype=torch.bool, device=problems.device)
            last_idx_flat = torch.zeros(batch * problem_size, dtype=torch.long, device=problems.device)
            for b in range(batch):
                seq = initial_tour[b]
                for t in range(problem_size):
                    pref = seq[:t+1]
                    visited_masks[b, t, pref] = True
                    last_idx_flat[b * problem_size + t] = pref[-1]
            # 展平为 (batch*problem, problem, 2) 以并行调用价值网络
            problems_flat = problems.unsqueeze(1).expand(batch, problem_size, problem_size, 2).reshape(batch * problem_size, problem_size, 2)
            visited_flat = visited_masks.reshape(batch * problem_size, problem_size)
            # 3) 预测“剩余长度” V(s_t)，并还原到 (batch, problem)
            pred_remaining = self.value_net(problems_flat, visited_flat, last_idx_flat).reshape(batch, problem_size)
            # 4) 价值引导的回溯潜力 φ = L_actual - V(s_t)
            phi = suffix_len - pred_remaining
            logits = phi
            # 5) 温度软化采样 top-k 回溯点
            chosen_points = sample_topk_from_logits(logits, k=self.k_backtrack, temperature=self.temperature)
            # 6) 编码节点并构造优化状态（用于后续前缀固定的后缀重构）
            encoded_nodes = self.encoder(problems)
            opt_state = OptimizationState(problems, initial_tour, encoded_nodes)
            # 7) 并行重构：对每个回溯点生成 m 个后缀候选（采样若干 + 贪心一次）
            candidates = [initial_tour]
            for t in range(self.k_backtrack):
                point = chosen_points[:, t]
                prefix_len = point + 1
                for mode in (['softmax'] * max(self.m_reconstruct - 1, 0) + (['argmax'] if self.m_reconstruct >= 1 else [])):
                    prefix = torch.stack([initial_tour[b, :prefix_len[b]] for b in range(batch)], dim=0)
                    tour_new, _ = self.reconstruct.reconstruct_suffix_state(problems, opt_state, prefix=prefix, eval_type=mode, return_logprob=False)
                    candidates.append(tour_new)
            # 8) 计算所有候选的路径长度，并用共享基线选择优势最大的候选
            lengths = [compute_tour_length(problems, c) for c in candidates]
            lengths_stack = torch.stack(lengths, dim=1)
            baseline = lengths_stack.mean(dim=1, keepdim=True)
            advantages = baseline - lengths_stack
            best_idx = torch.argmax(advantages, dim=1)
            cand_stack = torch.stack(candidates, dim=0)
            best_tours = cand_stack[best_idx, torch.arange(batch), :]
            best_lengths = lengths_stack[torch.arange(batch), best_idx]
        else:
            # 数据增强分支：把坐标做 8 倍增广，路径复制 8 份，随后同样流程
            problems_aug = augment_xy_data_by_8_fold(problems)
            initial_aug = initial_tour.repeat(8, 1)
            aug_batch = problems_aug.size(0)
            suffix_len = compute_suffix_lengths(problems_aug, initial_aug)
            visited_masks = torch.zeros(aug_batch, problem_size, problem_size, dtype=torch.bool, device=problems.device)
            last_idx_flat = torch.zeros(aug_batch * problem_size, dtype=torch.long, device=problems.device)
            for b in range(aug_batch):
                seq = initial_aug[b]
                for t in range(problem_size):
                    pref = seq[:t+1]
                    visited_masks[b, t, pref] = True
                    last_idx_flat[b * problem_size + t] = pref[-1]
            problems_flat = problems_aug.unsqueeze(1).expand(aug_batch, problem_size, problem_size, 2).reshape(aug_batch * problem_size, problem_size, 2)
            visited_flat = visited_masks.reshape(aug_batch * problem_size, problem_size)
            pred_remaining = self.value_net(problems_flat, visited_flat, last_idx_flat).reshape(aug_batch, problem_size)
            phi = suffix_len - pred_remaining
            logits = phi
            chosen_points = sample_topk_from_logits(logits, k=self.k_backtrack, temperature=self.temperature)
            encoded_nodes_aug = self.encoder(problems_aug)
            opt_state_aug = OptimizationState(problems_aug, initial_aug, encoded_nodes_aug)
            candidates = [initial_aug]
            for t in range(self.k_backtrack):
                point = chosen_points[:, t]
                prefix_len = point + 1
                for mode in (['softmax'] * max(self.m_reconstruct - 1, 0) + (['argmax'] if self.m_reconstruct >= 1 else [])):
                    prefix = torch.stack([initial_aug[b, :prefix_len[b]] for b in range(aug_batch)], dim=0)
                    tour_new, _ = self.reconstruct.reconstruct_suffix_state(problems_aug, opt_state_aug, prefix=prefix, eval_type=mode, return_logprob=False)
                    candidates.append(tour_new)
            # 按原批次维度从 8 份增广候选中选优势最大的路径
            cand_stack = torch.stack(candidates, dim=0)
            probs_rep = problems.repeat(8, 1, 1)
            lengths_stack = torch.stack([compute_tour_length(probs_rep, c) for c in candidates], dim=1)
            best_tours = torch.zeros(batch, problem_size, dtype=initial_tour.dtype, device=problems.device)
            best_lengths = torch.zeros(batch, dtype=problems.dtype, device=problems.device)
            for b in range(batch):
                idxs = (torch.arange(8, device=problems.device) * batch + b)
                sub_lengths = lengths_stack[idxs]
                baseline = sub_lengths.mean()
                advantages = baseline - sub_lengths
                flat_idx = torch.argmax(advantages)
                c_idx = flat_idx % sub_lengths.size(1)
                a_idx = idxs[flat_idx // sub_lengths.size(1)]
                best_tours[b] = cand_stack[c_idx, a_idx]
                best_lengths[b] = sub_lengths.view(-1)[flat_idx]

        # 返回结果与度量：包含初始长度与是否改进
        result = {
            'best_tours': best_tours,
            'best_lengths': best_lengths,
            'initial_lengths': compute_tour_length(problems, initial_tour),
            'improved': (best_lengths < compute_tour_length(problems, initial_tour)),
        }
        return result
