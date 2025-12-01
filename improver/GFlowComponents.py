import torch
import torch.nn as nn
import torch.nn.functional as F
from improver.ImproverModelParts import SharedEncoder, SharedDecoder, compute_tour_length

class OptimizationState:
    """优化状态：记录当前优化过程的状态
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
    """价值网络：预测从当前状态到终点的剩余路径长度
    输入：
      - problems: (batch, problem, 2)
      - visited_mask: (batch, problem) 访问过为1，未访问为0
      - last_idx: (batch,) 前缀最后节点索引
    输出：
      - pred_remaining: (batch,) 预测的剩余长度
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
        node_embed = self.encoder(problems)
        x = node_embed.transpose(1, 2)
        pooled = self.pool(x)
        global_ctx = pooled.squeeze(2)
        pred = self.head(global_ctx).squeeze(1)
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

    def reconstruct_suffix(self, problems: torch.Tensor, opt_state: OptimizationState, prefix: torch.Tensor,
                           eval_type: str = 'argmax') -> torch.Tensor:
        tour, _ = self.reconstruct_suffix_state(problems, opt_state, prefix=prefix, eval_type=eval_type, return_logprob=False)
        return tour

    def reconstruct_suffix_state(self, problems: torch.Tensor, opt_state: OptimizationState,
                                 prefix: torch.Tensor, eval_type: str = 'argmax', return_logprob: bool = False):
        device = problems.device
        batch, problem_size, _ = problems.shape

        self.decoder.set_kv(opt_state.encoded_nodes)
        last_idx = prefix[:, -1]
        q1_nodes = opt_state.encoded_nodes[torch.arange(batch), last_idx].unsqueeze(1)
        self.decoder.set_q1(q1_nodes)

        tour = prefix.clone()
        current_last = last_idx.clone()
        logprob_sum = torch.zeros(batch, device=device)
        visited = torch.zeros(batch, problem_size, dtype=torch.bool, device=device)
        visited[torch.arange(batch)[:, None], prefix] = True

        while tour.size(1) < problem_size:
            ninf_mask = torch.zeros((batch, 1, problem_size), device=device)
            ninf_mask = ninf_mask.masked_fill(visited.unsqueeze(1), float('-inf'))
            enc_last = opt_state.encoded_nodes[torch.arange(batch), current_last].unsqueeze(1)
            probs = self.decoder(enc_last, ninf_mask=ninf_mask)

            if eval_type == 'softmax':
                if (probs.sum(dim=2) == 0).any():
                    raise RuntimeError("All probabilities are zero under mask; check visited")
                logits = torch.log(probs.squeeze(1) + 1e-12)
                selected = torch.distributions.Categorical(logits=logits).sample()
                if return_logprob:
                    logprob_sum = logprob_sum + logits[torch.arange(batch), selected]
            else:
                selected = probs.argmax(dim=2).squeeze(1)

            tour = torch.cat([tour, selected[:, None]], dim=1)
            visited[torch.arange(batch), selected] = True
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
