import torch
import torch.nn as nn
import torch.nn.functional as F
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
        pooled = self.pool(x)                           # (batch, D, 1)
        global_ctx = pooled.squeeze(2)                  # (batch, D)
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

 

def _apply_insertion_to_tour(tour: torch.Tensor, prefix_len: int, city: int, insert_after_pos: int) -> torch.Tensor:
    """
    重构操作
    输入：
      - tour: (problem,) 当前构造的路径
      - prefix_len: 前缀长度 L
      - city: 未访问城市索引 u
      - insert_after_pos: 插入位置 e (当前前缀的边位置)
    输出：
      - new_tour: (problem,) 插入后的路径
    """
    problem = tour.size(0) # 城市总数
    # 定位待插入城市city在原路径中的位置
    pos = (tour == city).nonzero(as_tuple=False).squeeze(1)
    pos_u = int(pos.item())
    # 是否已访问城市u
    if pos_u < prefix_len:
        return tour.clone()
    
    insert_idx = max(prefix_len, min(insert_after_pos + 1, problem))
    
    if pos_u == insert_idx:
        return tour.clone()
    without = torch.cat([tour[:pos_u], tour[pos_u+1:]])
    left = without[:insert_idx]
    right = without[insert_idx:]
    return torch.cat([left, torch.tensor([city], dtype=tour.dtype, device=tour.device), right])

def sample_reconstruction_candidates_for_point(problems: torch.Tensor, tour: torch.Tensor, prefix_len: torch.Tensor, m: int, value_net: nn.Module, temperature: float = 1.0):
    """
    抽样重构点
    输入：
      - problems: (batch, problem, 2) 城市坐标
      - tour: (batch, problem) 当前构造的路径
      - prefix_len: (batch,) 前缀长度 L (已访问城市数)
      - value_net: 价值网络
      - m: 每个样本返回的候选路径数量
      - temperature: 温度参数 τ
    输出：
      - cand_tensor: (batch, m, problem) 每个样本的 m 个候选重构路径
      - logprob_rows: (batch, m) 对应候选路径的对数概率
    """
    device = problems.device
    batch = problems.size(0)
    problem = problems.size(1)
    
    candidates = []
    actions_rows = []
    
    # 枚举每个样本
    for b in range(batch):
        L = int(prefix_len[b].item())
        suffix_cities = tour[b, L:] # 未访问城市集合 U
        
        actions = []
        tours_b = []
        
        # 枚举所有插入动作
        for u in suffix_cities.tolist(): # 枚举城市
            for e_pos in range(L-1, problem-1): # 枚举边
                new_tour = _apply_insertion_to_tour(tour[b], L, int(u), e_pos)
                
                actions.append((u, e_pos))
                tours_b.append(new_tour)
        
        if len(tours_b) == 0:
            tours_b = [tour[b].clone()]
            actions = [(-1, -1)]
        
        tours_stack = torch.stack(tours_b) # 所有 s_new 的集合
        # 构造价值网络的输入
        problems_rep = problems[b].unsqueeze(0).expand(tours_stack.size(0), problem, 2)
        visited_mask = torch.ones(tours_stack.size(0), problem, dtype=torch.bool, device=device)
        last_idx = tours_stack[:, -1]
        # 计算重构潜力 φ_F = -V(s_new)
        v_pred = value_net(problems_rep, visited_mask, last_idx)
        phi = -v_pred
        probs = F.softmax(phi.unsqueeze(0) / max(temperature, 1e-6), dim=1)
        probs_idx = torch.multinomial(probs.squeeze(0), num_samples=min(m, len(tours_b)), replacement=False)
        chosen = tours_stack[probs_idx]
        candidates.append(chosen)
        chosen_log = torch.log(probs.squeeze(0)[probs_idx] + 1e-12)
        chosen_actions = torch.tensor([actions[i] for i in probs_idx.tolist()], device=device, dtype=torch.long)
        if b == 0:
            logprob_rows = chosen_log.unsqueeze(0)
            actions_rows = chosen_actions.unsqueeze(0)
        else:
            logprob_rows = torch.cat([logprob_rows, chosen_log.unsqueeze(0)], dim=0)
            actions_rows = torch.cat([actions_rows, chosen_actions.unsqueeze(0)], dim=0)

    cand_tensor = torch.stack(candidates, dim=0)
    return cand_tensor, logprob_rows, actions_rows
