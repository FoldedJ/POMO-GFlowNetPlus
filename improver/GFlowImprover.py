import torch
import torch.nn as nn
from typing import Dict, Any

from improver.ImproverModelParts import SharedEncoder, SharedDecoder, compute_tour_length
from improver.GFlowComponents import ValueNetwork, BacktrackPolicy, ReconstructionPolicy, sample_topk_from_logits, ReconstructionState


class GFlowImprover:
    """提升式优化器：一次迭代执行共同回溯与并行重构，返回优中选优的路径。
    与现有 POMO 模型兼容：仅使用坐标输入，不修改原文件。
    """
    def __init__(self, model_params: Dict[str, Any], k_backtrack: int = 3, m_reconstruct: int = 3, temperature: float = 1.0):
        self.encoder = SharedEncoder(**model_params)  # 共享编码器：坐标->嵌入
        self.decoder = SharedDecoder(**model_params)  # 共享解码器：生成下一城市分布
        self.value_net = ValueNetwork(self.encoder)   # 价值网络（可用于评估）
        self.backtrack = BacktrackPolicy(self.encoder)  # 回溯策略：位置打分
        self.reconstruct = ReconstructionPolicy(self.decoder)  # 重构策略：基于前缀生成后缀

        self.k_backtrack = k_backtrack
        self.m_reconstruct = m_reconstruct
        self.temperature = temperature

    @torch.no_grad()
    def improve_once(self, problems: torch.Tensor, initial_tour: torch.Tensor) -> Dict[str, Any]:
        """Inputs: (batch, problem, 2), (batch, problem) -> Returns dict with
        best_tours:(batch, problem), best_lengths:(batch,), initial_lengths:(batch,), improved:(batch,)"""
        device = problems.device
        if problems.dim() != 3 or problems.size(-1) != 2:
            raise ValueError("problems must be (batch, problem, 2)")
        if initial_tour.dim() != 2:
            raise ValueError("initial_tour must be (batch, problem)")
        batch, problem_size, _ = problems.shape
        if initial_tour.size(0) != batch or initial_tour.size(1) != problem_size:
            raise ValueError("initial_tour must match problems batch and problem_size")

        # 编码一次，全局复用：避免重复计算
        encoded_nodes = self.encoder(problems)

        # 共同回溯：为每个位置打分，并按温度抽样 top-k 回溯点
        logits = self.backtrack.forward_with_encoded(encoded_nodes, initial_tour)  # (batch, problem) 避免重复编码
        chosen_points = sample_topk_from_logits(logits, k=self.k_backtrack, temperature=self.temperature)  # (batch, k)

        # 候选集合：包含原路径与若干重构路径
        candidates = [initial_tour]

        # 并行重构：对每个回溯点，保留前缀并重构后缀（softmax若干条+argmax一条）
        for t in range(self.k_backtrack):
            point = chosen_points[:, t]  # (batch,)
            prefix_len = point + 1  # 保留至该点
            # 构造不同重构风格（softmax/argmax）
            for mode in (['softmax'] * max(self.m_reconstruct - 1, 0) + (['argmax'] if self.m_reconstruct >= 1 else [])):
                # 前缀切片
                prefix = torch.stack([initial_tour[b, :prefix_len[b]] for b in range(batch)], dim=0)
                state = ReconstructionState.from_prefix(encoded_nodes, prefix)
                tour_new, _ = self.reconstruct.reconstruct_suffix_state(problems, state, prefix=prefix, eval_type=mode, return_logprob=False)
                candidates.append(tour_new)

        # 评估所有候选长度，选择最短路径作为改进结果
        lengths = [compute_tour_length(problems, c) for c in candidates]
        lengths_stack = torch.stack(lengths, dim=1)  # (batch, num_candidates)
        best_idx = torch.argmin(lengths_stack, dim=1)  # (batch,)
        cand_stack = torch.stack(candidates, dim=0)  # (C, batch, problem)
        best_tours = cand_stack[best_idx, torch.arange(batch), :]  # (batch, problem)
        best_lengths = lengths_stack[torch.arange(batch), best_idx]

        # 返回结果与度量：包含初始长度与是否改进
        result = {
            'best_tours': best_tours,
            'best_lengths': best_lengths,
            'initial_lengths': compute_tour_length(problems, initial_tour),
            'improved': (best_lengths < compute_tour_length(problems, initial_tour)),
        }
        return result
