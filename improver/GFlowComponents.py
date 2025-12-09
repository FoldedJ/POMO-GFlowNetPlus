import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from improver.ImproverModelParts import SharedEncoder, compute_suffix_lengths

 

class ValueNetwork(nn.Module):
    """
    价值网络
    输入：
      problems: (batch, problem, 2) 城市坐标
      tour: (batch, problem) 路径
      mask: (batch, problem) 掩码
    输出：
      pred: (batch,) 价值预测
    """
    def __init__(self, shared_encoder: SharedEncoder):
        super().__init__()
        self.encoder = shared_encoder
        D = self.encoder.model_params['embedding_dim']
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, problems: torch.Tensor, tour: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, problem_size, _ = problems.shape
        # 构造含坐标的路径信息
        gather_idx = tour.unsqueeze(2).expand(batch, problem_size, 2)
        ordered = problems.gather(dim=1, index=gather_idx)
        # 编码路径信息
        base_emb = self.encoder.embedding(ordered)
        # 构造位置编码
        dim = self.encoder.model_params['embedding_dim']
        position = torch.arange(problem_size, device=problems.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=problems.device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(problem_size, dim, device=problems.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pe.unsqueeze(0).expand(batch, problem_size, dim)
        # 合并路径信息和位置编码
        node_embed = base_emb + pos_embed
        
        if mask is not None:
            rank3_mask = (~mask).float().unsqueeze(1).expand(batch, problem_size, problem_size) * (-1e9)
            node_embed = self.encoder.encode_from_embeddings(node_embed, rank3_ninf_mask=rank3_mask)
        else:
            node_embed = self.encoder.encode_from_embeddings(node_embed)
        x = node_embed.transpose(1, 2)
        pooled = self.pool(x)
        global_ctx = pooled.squeeze(2)
        pred = self.head(global_ctx).squeeze(1)
        return pred

def sample_backtrack_points(problems: torch.Tensor, tour: torch.Tensor, value_net: nn.Module, k: int, temperature: float = 1.0):
    """回溯
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
    
    # 真实后缀长度 L_actual(π[i:N])
    suffix_len = compute_suffix_lengths(problems, tour)
    # node_pos[b, city] = city 在 tour[b] 中的位置
    node_pos = torch.argsort(tour, dim=1)
    # 构造时间步张量与位置扩展
    t_range = torch.arange(problem_size, device=device).view(1, problem_size, 1)
    pos_expanded = node_pos.unsqueeze(1)
    # 生成后缀掩码：仅保留从当前位置到终点的子路径，当前位置之前的点全部屏蔽
    suffix_masks = (pos_expanded >= t_range)
    
    # 扁平化处理
    problems_flat = problems.unsqueeze(1).expand(batch, problem_size, problem_size, 2).reshape(batch * problem_size, problem_size, 2)
    mask_flat = suffix_masks.reshape(batch * problem_size, problem_size)
    tour_flat = tour.unsqueeze(1).expand(batch, problem_size, problem_size).reshape(batch * problem_size, problem_size)
    
    # 价值网络预测剩余长度
    pred_remaining = value_net(problems_flat, tour=tour_flat, mask=mask_flat).reshape(batch, problem_size)
    # 计算回溯潜力 φ_B(i) = L_actual(π[i:N]) - V(s_i)
    phi = suffix_len - pred_remaining
    # 将最后一个点概率置零后重归一化
    logits = phi / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=1)
    mask = torch.ones_like(probs)
    mask[:, -1] = 0.0
    probs = probs * mask
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    _, idxs = torch.topk(probs, k=k, dim=1, largest=True, sorted=True)
    log_probs = torch.log(probs + 1e-12)
    return idxs, log_probs

def _apply_edge_split_reinsertion(tour: torch.Tensor, backtrack_pos: int, edge_idx: int) -> torch.Tensor:
    """移除回溯点并通过拆分环路中的一条边将该点重新插入。
    tour: (problem,)
    backtrack_pos: 被移除的节点在 tour 中的位置
    edge_idx: 在移除后环路中的边索引 [0..problem-2]，对应 (u,v) 为 (cycle[edge_idx], cycle[(edge_idx+1)%len])
    返回: 新的包含该节点的 tour
    """
    problem = tour.size(0)
    node = int(tour[backtrack_pos].item())
    # 构造移除后的环路
    cycle = torch.cat([tour[:backtrack_pos], tour[backtrack_pos+1:]])  # (problem-1)
    L = cycle.size(0)
    u_idx = int(edge_idx)
    if u_idx < 0 or u_idx >= L:
        u_idx = max(0, min(u_idx, L-1))
    # 在 u 之后插入该节点
    left = cycle[:u_idx+1]
    right = cycle[u_idx+1:]
    return torch.cat([left, torch.tensor([node], dtype=tour.dtype, device=tour.device), right])

def sample_reconstruction_by_edge_split(problems: torch.Tensor, tour: torch.Tensor, backtrack_pos_batch: torch.Tensor, m: int, value_net: nn.Module, temperature: float = 1.0):
    """重构
    输入：
      - problems: (batch, problem, 2)
      - tour: (batch, problem)
      - backtrack_pos_batch: (batch,) 选择的回溯点位置
      - m: 候选数
    输出：
      - cand_tensor: (batch, m, problem)
      - logprob_rows: (batch, m)
      - actions_rows: (batch, m) 边索引（0..problem-2）
    """
    device = problems.device
    batch, problem, _ = problems.shape
    
    candidates = [] # 候选路径
    logprob_rows = None # 候选路径的对数概率
    actions_rows = None # 对应边索引
    
    for b in range(batch):
        bp = int(backtrack_pos_batch[b].item()) # 回溯点
        # 枚举 (N-1) 条边（移除回溯点后的环路）
        tours_b = []
        actions_b = []
        for e_idx in range(problem-1):
            new_t = _apply_edge_split_reinsertion(tour[b], bp, e_idx)
            tours_b.append(new_t)
            actions_b.append(e_idx)
        tours_stack = torch.stack(tours_b)  # (problem-1, problem)
        # 评分与抽样
        problems_rep = problems[b].unsqueeze(0).expand(tours_stack.size(0), problem, 2)
        visited_mask = None
        # 价值网络预测
        v_pred = value_net(problems_rep, tour=tours_stack, mask=visited_mask)
        # 计算重构潜力 φ = -V(s_new)
        phi = -v_pred
        probs = F.softmax(phi / max(temperature, 1e-6), dim=0)
        # 选择重构路径
        num = min(m, tours_stack.size(0))
        _, chosen_idx = torch.topk(probs, k=num, dim=0, largest=True, sorted=True)
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
