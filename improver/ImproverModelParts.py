import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    """SharedEncoder
    forward(problems: Tensor) -> Tensor
    - Input: (batch, problem, 2)
    - Output: (batch, problem, embedding_dim)
    - Errors: ValueError for invalid shapes
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)  # 将坐标(2维)映射到嵌入空间
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])  # 堆叠若干编码层

    def forward(self, data):
        """Input: (batch, problem, 2) -> Output: (batch, problem, embedding_dim)"""
        if data.dim() != 3 or data.size(-1) != 2:
            raise ValueError("problems tensor must be (batch, problem, 2)")
        embedded_input = self.embedding(data)  # 逐城市坐标编码
        out = embedded_input
        for layer in self.layers:  # 逐层进行自注意力与前馈
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 多头Q
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 多头K
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 多头V
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)  # 合并多头输出

        self.addAndNormalization1 = AddAndNorm(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = AddAndNorm(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)  # 自注意力
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.addAndNormalization1(input1, multi_head_out)  # 残差+归一化
        out2 = self.feedForward(out1)  # 前馈网络
        out3 = self.addAndNormalization2(out1, out2)  # 残差+归一化
        return out3


class SharedDecoder(nn.Module):
    """SharedDecoder
    set_kv(encoded_nodes), set_q1(encoded_q1), forward(encoded_last_node, ninf_mask)
    - Shapes遵循接口约定，异常时抛出 ValueError
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 首节点Q
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)   # 末节点Q
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)       # K
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)       # V
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)   # 合并多头

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

    def set_kv(self, encoded_nodes):
        """Input: (batch, problem, embedding) -> prepare K,V, single_head_key"""
        head_num = self.model_params['head_num']
        if encoded_nodes.dim() != 3:
            raise ValueError("encoded_nodes must be (batch, problem, embedding)")
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)  # (batch, head, n, dim)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)  # (batch, emb, n) 用于打分

    def set_q1(self, encoded_q1):
        """Input: (batch, n, embedding) -> prepare q_first"""
        head_num = self.model_params['head_num']
        if encoded_q1.dim() != 3:
            raise ValueError("encoded_q1 must be (batch, n, embedding)")
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)  # (batch, head, n, dim)

    def forward(self, encoded_last_node, ninf_mask):
        """Inputs: (batch, n, embedding), (batch, group, problem) -> Output: (batch, n, problem)"""
        head_num = self.model_params['head_num']
        if encoded_last_node.dim() != 3:
            raise ValueError("encoded_last_node must be (batch, n, embedding)")
        if ninf_mask is None or ninf_mask.dim() != 3:
            raise ValueError("ninf_mask must be (batch, group, problem)")
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)  # 末节点Q
        q = self.q_first + q_last  # 组合首末节点信息
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)  # 掩码注意力
        mh_atten_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_atten_out, self.single_head_key)  # 与K(单头)做相似度打分

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim  # 缩放避免梯度爆炸
        score_clipped = logit_clipping * torch.tanh(score_scaled)  # logits裁剪
        score_masked = score_clipped + ninf_mask  # 已访问位置置为 -inf
        probs = F.softmax(score_masked, dim=2)  # 得到选择分布
        return probs


def reshape_by_heads(qkv, head_num):
    """Reshape linear outputs to (batch, head, n, dim_per_head) for multi-head attention"""
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)  # 切分成多头
    q_transposed = q_reshaped.transpose(1, 2)  # (batch, head, n, dim)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """Scaled dot-product attention with optional -inf masks; returns concatenated heads"""
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # 计算注意力打分
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))  # 缩放
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)  # 掩码
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)  # 掩码
    weights = nn.Softmax(dim=3)(score_scaled)  # 归一化得到权重
    out = torch.matmul(weights, v)  # 聚合V
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # 合并多头
    return out_concat


class AddAndNorm(nn.Module):
    """Residual add + InstanceNorm; keeps embedding_dim; no shape change except norm axes"""
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2  # 残差相加
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)  # 通道维归一化
        back_trans = normalized.transpose(1, 2)
        return back_trans


class FeedForward(nn.Module):
    """Position-wise FFN: Linear(emb->hidden)->ReLU->Linear(hidden->emb)"""
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))  # 逐位置前馈


def compute_tour_length(problems: torch.Tensor, tours: torch.Tensor) -> torch.Tensor:
    """计算路径长度
    problems: (batch, problem, 2)
    tours: (batch, problem) 每个元素为城市索引
    返回: (batch,) 总长度
    """
    if problems.dim() != 3 or problems.size(-1) != 2:
        raise ValueError("problems tensor must be (batch, problem, 2)")
    if tours.dim() != 2:
        raise ValueError("tours tensor must be (batch, problem)")
    batch = problems.size(0)
    problem_size = problems.size(1)
    if tours.size(0) != batch or tours.size(1) != problem_size:
        raise ValueError("tours must match problems batch and problem_size")
    if tours.dtype != torch.long:
        tours = tours.long()
    if (tours < 0).any() or (tours >= problem_size).any():
        raise ValueError("tours indices out of range")
    gathering_index = tours.unsqueeze(2).expand(batch, problem_size, 2)
    ordered_seq = problems.gather(dim=1, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
    travel_distances = segment_lengths.sum(1)
    return travel_distances

def compute_remaining_length(problems: torch.Tensor, tours: torch.Tensor, prefix_len: torch.Tensor) -> torch.Tensor:
    """计算剩余路径长度
    problems: (batch, problem, 2)
    tours: (batch, problem) 每个元素为城市索引
    prefix_len: (batch,) 每个样本的已访问城市数
    返回: (batch,) 剩余路径长度
    """
    if problems.dim() != 3 or problems.size(-1) != 2:
        raise ValueError("problems tensor must be (batch, problem, 2)")
    if tours.dim() != 2:
        raise ValueError("tours tensor must be (batch, problem)")
    if prefix_len.dim() != 1 or prefix_len.size(0) != problems.size(0):
        raise ValueError("prefix_len must be (batch,)")
    batch = problems.size(0)
    problem_size = problems.size(1)
    if tours.size(0) != batch or tours.size(1) != problem_size:
        raise ValueError("tours must match problems batch and problem_size")
    if tours.dtype != torch.long:
        tours = tours.long()
    gathering_index = tours.unsqueeze(2).expand(batch, problem_size, 2)
    ordered_seq = problems.gather(dim=1, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
    start_idx = (prefix_len - 1).clamp(min=0)
    idx = torch.arange(problem_size, device=problems.device).unsqueeze(0).expand(batch, problem_size)
    mask = (idx >= start_idx.unsqueeze(1)).float()
    return (segment_lengths * mask).sum(1)

def compute_suffix_lengths(problems: torch.Tensor, tours: torch.Tensor) -> torch.Tensor:
    """计算后缀路径长度
    problems: (batch, problem, 2)
    tours: (batch, problem) 每个元素为城市索引
    返回: (batch, problem) 每个城市到路径结束的距离
    """
    if problems.dim() != 3 or problems.size(-1) != 2:
        raise ValueError("problems tensor must be (batch, problem, 2)")
    if tours.dim() != 2:
        raise ValueError("tours tensor must be (batch, problem)")
    batch = problems.size(0)
    problem_size = problems.size(1)
    if tours.size(0) != batch or tours.size(1) != problem_size:
        raise ValueError("tours must match problems batch and problem_size")
    if tours.dtype != torch.long:
        tours = tours.long()
    gathering_index = tours.unsqueeze(2).expand(batch, problem_size, 2)
    ordered_seq = problems.gather(dim=1, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()  # (batch, problem)
    suffix = torch.zeros_like(segment_lengths)
    for t in range(problem_size):
        suffix[:, t] = segment_lengths[:, t:].sum(dim=1)
    return suffix

