# 用于推理
import torch
import torch.nn as nn
from typing import Dict, Any

from improver.ImproverModelParts import SharedEncoder, compute_tour_length, compute_suffix_lengths
from improver.GFlowComponents import ValueNetwork, OptimizationState, sample_backtrack_points, sample_reconstruction_candidates_for_point
from TSProblemDef import augment_xy_data_by_8_fold


class GFlowImprover:
    """提升式优化器：一次迭代执行共同回溯与并行重构，返回优中选优的路径。
    """
    def __init__(self, model_params: Dict[str, Any], k_backtrack: int = 3, m_reconstruct: int = 3, temperature: float = 1.0, use_augmentation: bool = False):
        self.encoder = SharedEncoder(**model_params)
        self.value_net = ValueNetwork(self.encoder)

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
            # 使用封装函数 sample_backtrack_points 进行回溯点选择
            chosen_points = sample_backtrack_points(problems, initial_tour, self.value_net, self.k_backtrack, self.temperature)
            
            candidates = [initial_tour]
            for t in range(self.k_backtrack):
                point = chosen_points[:, t]
                prefix_len = point + 1
                cand, _ = sample_reconstruction_candidates_for_point(problems, initial_tour, prefix_len, self.m_reconstruct, self.value_net, self.temperature)
                for i in range(cand.size(1)):
                    candidates.append(cand[:, i, :])
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
            
            chosen_points = sample_backtrack_points(problems_aug, initial_aug, self.value_net, self.k_backtrack, self.temperature)

            candidates = [initial_aug]
            for t in range(self.k_backtrack):
                point = chosen_points[:, t]
                prefix_len = point + 1
                cand, _ = sample_reconstruction_candidates_for_point(problems_aug, initial_aug, prefix_len, self.m_reconstruct, self.value_net, self.temperature)
                for i in range(cand.size(1)):
                    candidates.append(cand[:, i, :])
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
