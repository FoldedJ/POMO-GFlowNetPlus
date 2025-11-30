import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from improver.ImproverModelParts import SharedEncoder, SharedDecoder, compute_tour_length


@dataclass
class ReconstructionState:   
    """重构状态，供重构使用
    - encoded_nodes: (batch, problem, D)
    - visited: (batch, problem) bool mask
    """
    encoded_nodes: torch.Tensor # 编码后的节点特征
    visited: torch.Tensor # 访问掩码

    @classmethod
    def from_prefix(cls, encoded_nodes: torch.Tensor, prefix: torch.Tensor):
        """根据前缀构造重构状态（初始化访问掩码）
        Args:
            encoded_nodes: (batch, problem, D) 编码后的节点特征
            prefix: (batch, k) 前缀路径，k为前缀长度
        Returns:
            ReconstructionState: 重构状态
        """
        batch, problem_size, _ = encoded_nodes.shape
        device = encoded_nodes.device
        visited = torch.zeros((batch, problem_size), dtype=torch.bool, device=device)  # 初始掩码
        if prefix.numel() > 0:
            visited[torch.arange(batch)[:, None], prefix] = True
        return cls(encoded_nodes=encoded_nodes, visited=visited)

class ValueNetwork(nn.Module):
    """价值网络：预测从当前状态到终点的剩余路径长度
    输入：
      - problems: (batch, problem, 2)
      - visited_mask: (batch, problem) 访问过为1，未访问为0
      - last_idx: (batch,) 前缀最后节点索引
    输出：
      - pred_remaining: (batch,) 预测的剩余长度
    """
    def __init__(self, shared_encoder: SharedEncoder, hidden_dims=(128, 64)):
        super().__init__()
        self.encoder = shared_encoder
        # 拼接：最后节点嵌入、全局上下文、剩余城市数、前缀长度
        in_dim = self.encoder.model_params['embedding_dim'] * 2 + 2
        dims = [in_dim] + list(hidden_dims) + [1]
        layers = []
        # 构造价值网络（线性层 + ReLU 激活）
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.head = nn.Sequential(*layers)

    def forward(self, problems: torch.Tensor, visited_mask: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
        if problems.dim() != 3 or problems.size(-1) != 2:
            raise ValueError("problems must be (batch, problem, 2)")
        if visited_mask.dim() != 2:
            raise ValueError("visited_mask must be (batch, problem)")
        
        batch, problem_size, _ = problems.shape
        node_embed = self.encoder(problems)  # (batch, problem, D) 图编码
        D = node_embed.size(2)
        
        if last_idx.dim() != 1 or last_idx.size(0) != batch:
            raise ValueError("last_idx must be (batch,)")
        
        last_embed = node_embed[torch.arange(batch), last_idx]  # (batch, D) 前缀最后节点嵌入
        global_ctx = node_embed.mean(dim=1)  # (batch, D) 全局平均池化
        remaining_count = (visited_mask == 0).sum(dim=1).float().unsqueeze(1)  # (batch, 1) 剩余城市数
        prefix_count = (visited_mask == 1).sum(dim=1).float().unsqueeze(1)  # (batch, 1) 前缀长度

        feat = torch.cat([last_embed, global_ctx, remaining_count, prefix_count], dim=1)  # 融合状态特征
        pred = self.head(feat).squeeze(1)  # (batch,) 预测剩余长度
        return pred


class BacktrackPolicy(nn.Module):
    """回溯策略：为每个位置输出回溯潜力 logits
    输入：
      - problems: (batch, problem, 2)
      - tour: (batch, problem)
    输出：
      - logits: (batch, problem)
    """
    def __init__(self, shared_encoder: SharedEncoder):
        super().__init__()
        self.encoder = shared_encoder
        D = self.encoder.model_params['embedding_dim']
        self.head = nn.Sequential(
            nn.Linear(D * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, problems: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
        if problems.dim() != 3 or problems.size(-1) != 2:
            raise ValueError("problems must be (batch, problem, 2)")
        if tour.dim() != 2:
            raise ValueError("tour must be (batch, problem)")
        batch, problem_size, _ = problems.shape
        node_embed = self.encoder(problems)  # (batch, problem, D)
        global_ctx = node_embed.mean(dim=1).unsqueeze(1).expand(batch, problem_size, node_embed.size(2))  # 平均作为上下文
        x = torch.cat([node_embed, global_ctx], dim=2)  # 拼接节点与全局
        logits = self.head(x).squeeze(2)  # (batch, problem) 每位置回溯潜力
        return logits

    def forward_with_encoded(self, encoded_nodes: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
        """使用已计算好的节点嵌入，避免重复编码 (batch, problem, D) -> logits(batch, problem)
        参数中的 tour 仅用于检查维度一致性，不参与计算
        """
        if encoded_nodes.dim() != 3:
            raise ValueError("encoded_nodes must be (batch, problem, D)")
        if tour.dim() != 2:
            raise ValueError("tour must be (batch, problem)")
        batch, problem_size, D = encoded_nodes.shape
        if tour.size(0) != batch or tour.size(1) != problem_size:
            raise ValueError("tour must match encoded_nodes shape")
        global_ctx = encoded_nodes.mean(dim=1).unsqueeze(1).expand(batch, problem_size, D)
        x = torch.cat([encoded_nodes, global_ctx], dim=2)
        logits = self.head(x).squeeze(2)
        return logits




class ReconstructionPolicy(nn.Module):
    """重构策略：在前缀固定的情况下，用解码器生成剩余路径的概率并取样或取argmax。
    使用 SharedDecoder，逐步扩展直到覆盖所有城市。
    """
    def __init__(self, shared_decoder: SharedDecoder):
        super().__init__()
        self.decoder = shared_decoder

    

    def reconstruct_suffix(self, problems: torch.Tensor, encoded_nodes: torch.Tensor, prefix: torch.Tensor,
                           eval_type: str = 'argmax') -> torch.Tensor:
        """基于前缀生成剩余路径，返回完整路径 (batch, problem)"""
        state = ReconstructionState.from_prefix(encoded_nodes, prefix)  # 构造显式状态
        tour, _ = self.reconstruct_suffix_state(problems, state, prefix=prefix, eval_type=eval_type, return_logprob=False)
        return tour

    def reconstruct_suffix_state(self, problems: torch.Tensor, state: ReconstructionState,
                                 prefix: torch.Tensor, eval_type: str = 'argmax', return_logprob: bool = False):
        """基于显式 state 生成后缀
        Inputs:
          - problems: (batch, problem, 2)
          - state: ReconstructionState(encoded_nodes, visited)
          - prefix: (batch, t) 当前已固定的前缀序列
        Returns:
          - tour: (batch, problem)
          - logprob_sum: (batch,) if return_logprob, else zeros
        """
        device = problems.device
        batch, problem_size, _ = problems.shape

        self.decoder.set_kv(state.encoded_nodes)  # 设定K/V
        last_idx = prefix[:, -1]
        q1_nodes = state.encoded_nodes[torch.arange(batch), last_idx].unsqueeze(1)  # 首节点q
        self.decoder.set_q1(q1_nodes)

        tour = prefix.clone()
        current_last = last_idx.clone()
        logprob_sum = torch.zeros(batch, device=device)  # 训练时累计logP

        while tour.size(1) < problem_size:
            ninf_mask = torch.zeros((batch, 1, problem_size), device=device)  # 掩码：已访问不可选
            ninf_mask = ninf_mask.masked_fill(state.visited.unsqueeze(1), float('-inf'))
            enc_last = state.encoded_nodes[torch.arange(batch), current_last].unsqueeze(1)  # (batch,1,D)
            probs = self.decoder(enc_last, ninf_mask=ninf_mask)  # (batch,1,problem)

            if eval_type == 'softmax':
                if (probs.sum(dim=2) == 0).any():
                    raise RuntimeError("All probabilities are zero under mask; check state/visited")
                logits = torch.log(probs.squeeze(1) + 1e-12)  # 数值稳定
                selected = torch.distributions.Categorical(logits=logits).sample()  # 抽样选择下一城市
                if return_logprob:
                    logprob_sum = logprob_sum + logits[torch.arange(batch), selected]
            else:
                selected = probs.argmax(dim=2).squeeze(1)  # 贪心选择

            tour = torch.cat([tour, selected[:, None]], dim=1)  # 扩展路径
            state.visited[torch.arange(batch), selected] = True  # 更新状态
            current_last = selected

        return tour, logprob_sum


def sample_topk_from_logits(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """对每行 logits 进行 softmax(含温度) 后抽样 top-k 不重复索引。
    返回形状 (batch, k)
    """
    if logits.dim() != 2:
        raise ValueError("logits must be (batch, problem)")
    if k < 1:
        raise ValueError("k must be >= 1")
    probs = F.softmax(logits / max(temperature, 1e-6), dim=1)  # 温度软化
    batch = logits.size(0)
    # 使用重复抽样后去重/补齐的简化方案
    sampled = probs.multinomial(num_samples=k, replacement=True)  # (batch, k)
    # 去重：简单策略为排序后唯一；若不足则补充最高概率项
    uniq = []
    for b in range(batch):
        row = sampled[b].tolist()
        seen = []
        for x in row:
            if x not in seen:
                seen.append(x)
        if len(seen) < k:
            topk = torch.topk(probs[b], k=k).indices.tolist()  # 最高概率用于补齐
            for x in topk:
                if x not in seen:
                    seen.append(x)
                if len(seen) == k:
                    break
        uniq.append(torch.tensor(seen[:k], dtype=torch.long))
    return torch.stack(uniq, dim=0)
