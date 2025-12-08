import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from improver.ImproverModelParts import SharedEncoder, compute_tour_length, compute_suffix_lengths

class OptimizationState:
    """
    优化状态：记录当前优化过程的状态
    属性：
      - problems: (batch, problem, 2) 问题实例
      - current_tour: (batch, problem) 当前构造的路径
      - encoded_nodes: (batch, problem, embedding_dim) 编码后的节点表示
      - value_net_params: 价值网络参数
      - step: 当前优化步数
    """
    def __init__(self, problems: torch.Tensor, current_tour: torch.Tensor, encoded_nodes: torch.Tensor,
                 value_net_params=None, step: int = 0):
        self.problems = problems
        self.current_tour = current_tour
        self.encoded_nodes = encoded_nodes
        self.value_net_params = value_net_params
        self.step = step

class ValueNetwork(nn.Module):
    """
    价值网络：估计从当前状态到终点的“剩余路径长度”V(s)
    输入：
      - problems: (batch, problem, 2) 城市坐标
      - visited_mask: (batch, problem) 序列前缀的访问掩码（访问=1，未访问=0）
      - last_idx: (batch,) 前缀最后一个城市索引
    输出：
      - pred_remaining: (batch,) 预测剩余路径长度 V(s)
    """
    def __init__(self, shared_encoder: SharedEncoder):
        super().__init__()
        self.encoder = shared_encoder
        D = self.encoder.model_params['embedding_dim']
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, problems: torch.Tensor, visited_mask: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
        batch, problem_size, _ = problems.shape
        node_embed = self.encoder(problems)              # (batch, N, D)
        x = node_embed.transpose(1, 2)                  # (batch, D, N)
        pooled = self.pool(x)
        global_ctx = pooled.squeeze(2)
        dim = self.encoder.model_params['embedding_dim']
        div_term = torch.exp(torch.arange(0, dim, 2, device=problems.device) * (-math.log(10000.0) / dim))
        step_ids = visited_mask.sum(dim=1).long().clamp(min=0, max=problem_size)
        step_angles = step_ids.unsqueeze(1) * div_term
        step_ctx = torch.zeros(batch, dim, device=problems.device)
        step_ctx[:, 0::2] = torch.sin(step_angles)
        step_ctx[:, 1::2] = torch.cos(step_angles)
        last_ids = last_idx.long().clamp(min=0, max=problem_size-1)
        last_angles = last_ids.unsqueeze(1) * div_term
        last_ctx = torch.zeros(batch, dim, device=problems.device)
        last_ctx[:, 0::2] = torch.sin(last_angles)
        last_ctx[:, 1::2] = torch.cos(last_angles)
        global_ctx = global_ctx + step_ctx + last_ctx
        pred = self.head(global_ctx).squeeze(1)         # (batch,)
        return pred

def sample_backtrack_points(problems: torch.Tensor, tour: torch.Tensor, value_net: nn.Module, k: int, temperature: float = 1.0):
    """
    抽样回溯点
    输入：
      - problems: (batch, problem, 2) 城市坐标
      - tour: (batch, problem) 当前构造的路径
      - value_net: 价值网络
      - k: 选择的回溯点数量
      - temperature: 温度参数 τ
    输出：
      - idx: (batch, k) 抽样得到的回溯点索引
      - log_probs: (batch, k) 对应回溯点的对数概率
    """
    batch, problem_size, _ = problems.shape
    device = problems.device
    
    # 计算真实的后缀长度（当前前缀之后到终点的总长度）
    suffix_len = compute_suffix_lengths(problems, tour)
    # node_pos[b, city] = city 在 tour[b] 中的位置
    node_pos = torch.argsort(tour, dim=1)
    # 构造时间步张量
    t_range = torch.arange(problem_size, device=device).view(1, problem_size, 1)
    pos_expanded = node_pos.unsqueeze(1)
    # 生成掩码（每一步状态的掩码）
    visited_masks = (pos_expanded <= t_range)
    
    # 扁平化处理
    last_idx_flat = tour.reshape(-1)
    problems_flat = problems.unsqueeze(1).expand(batch, problem_size, problem_size, 2).reshape(batch * problem_size, problem_size, 2)
    visited_flat = visited_masks.reshape(batch * problem_size, problem_size)
    
    # 价值网络预测剩余长度
    pred_remaining = value_net(problems_flat, visited_flat, last_idx_flat).reshape(batch, problem_size)
    # 计算回溯潜力 φ = 后缀真实长度 - 价值估计
    phi = suffix_len - pred_remaining
    probs = F.softmax(phi / max(temperature, 1e-6), dim=1)
    idxs = torch.multinomial(probs, num_samples=k, replacement=False)
    log_probs = torch.log(probs + 1e-12)
    return idxs, log_probs

 


def _apply_edge_split_reinsertion(tour: torch.Tensor, backtrack_pos: int, edge_idx: int) -> torch.Tensor:
    """移除回溯点并通过拆分环路中的一条边将该点重新插入。
    tour: (problem,)
    backtrack_pos: 被移除的节点在 tour 中的位置
    edge_idx: 在移除后环路中的边索引 [0..problem-2]，对应 (u,v) 为 (cycle[edge_idx], cycle[(edge_idx+1)%len])
    返回: 新的包含该节点的 tour
    """
    N = tour.size(0)
    node = int(tour[backtrack_pos].item())
    # 构造移除后的环路
    cycle = torch.cat([tour[:backtrack_pos], tour[backtrack_pos+1:]])  # (N-1)
    L = cycle.size(0)
    u_idx = int(edge_idx)
    if u_idx < 0 or u_idx >= L:
        u_idx = max(0, min(u_idx, L-1))
    # 在 u 之后插入该节点
    left = cycle[:u_idx+1]
    right = cycle[u_idx+1:]
    return torch.cat([left, torch.tensor([node], dtype=tour.dtype, device=tour.device), right])

def sample_reconstruction_by_edge_split(problems: torch.Tensor,
                                        tour: torch.Tensor,
                                        backtrack_pos_batch: torch.Tensor,
                                        m: int,
                                        value_net: nn.Module,
                                        temperature: float = 1.0):
    """按照“拆边重插”策略为每个样本生成 m 条候选路径。
    输入：
      - problems: (batch, N, 2)
      - tour: (batch, N)
      - backtrack_pos_batch: (batch,) 选择的回溯点位置
      - m: 候选数
    输出：
      - cand_tensor: (batch, m, N)
      - logprob_rows: (batch, m)
      - actions_rows: (batch, m) 边索引（0..N-2）
    """
    device = problems.device
    batch = problems.size(0)
    N = problems.size(1)
    candidates = []
    logprob_rows = None
    actions_rows = None
    for b in range(batch):
        bp = int(backtrack_pos_batch[b].item())
        # 枚举 (N-1) 条边（移除后的环路）
        tours_b = []
        actions_b = []
        for e_idx in range(N-1):
            new_t = _apply_edge_split_reinsertion(tour[b], bp, e_idx)
            tours_b.append(new_t)
            actions_b.append(e_idx)
        tours_stack = torch.stack(tours_b)  # (N-1, N)
        # 评分与抽样
        problems_rep = problems[b].unsqueeze(0).expand(tours_stack.size(0), N, 2)
        visited_mask = torch.ones(tours_stack.size(0), N, dtype=torch.bool, device=device)
        last_idx = tours_stack[:, -1]
        v_pred = value_net(problems_rep, visited_mask, last_idx)
        phi = -v_pred
        probs = F.softmax(phi / max(temperature, 1e-6), dim=0)
        num = min(m, tours_stack.size(0))
        chosen_idx = torch.multinomial(probs, num_samples=num, replacement=False)
        chosen_tours = tours_stack[chosen_idx]
        candidates.append(chosen_tours)
        chosen_log = torch.log(probs[chosen_idx] + 1e-12)
        chosen_actions = torch.tensor([actions_b[i] for i in chosen_idx.tolist()], device=device, dtype=torch.long)
        if b == 0:
            logprob_rows = chosen_log.unsqueeze(0)
            actions_rows = chosen_actions.unsqueeze(0)
        else:
            logprob_rows = torch.cat([logprob_rows, chosen_log.unsqueeze(0)], dim=0)
            actions_rows = torch.cat([actions_rows, chosen_actions.unsqueeze(0)], dim=0)
    cand_tensor = torch.stack(candidates, dim=0)
    return cand_tensor, logprob_rows, actions_rows
