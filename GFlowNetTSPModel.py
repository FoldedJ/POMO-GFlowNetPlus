import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence

from TSPModel import TSP_Decoder


# 辅助函数和模块 (从TSPModel.py复制)

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n 可以是 1 或 PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n 可以是 1 或 PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        # 需要输入的shape: (batch_size, num_features, length) 所以后续需要转置

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, problem, embedding_dim)
        # pe shape: (1, max_len, embedding_dim)
        return x + self.pe[:, :x.size(1)]


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=model_params['problem_size'])
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, problem, 2)

        # 输入归一化
        min_coords = data.min(dim=1, keepdim=True)[0] # (batch, 1, 2)
        max_coords = data.max(dim=1, keepdim=True)[0] # (batch, 1, 2)
        normalized_data = (data - min_coords) / (max_coords - min_coords + 1e-6) # 避免除以零

        embedded_input = self.embedding(normalized_data)
        embedded_input = self.positional_encoding(embedded_input)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


# 2. 价值网络架构
class ValueNetwork(nn.Module):
    def __init__(self, encoder: TSP_Encoder, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = encoder  # 复用POMO的编码器
        embedding_dim = self.model_params['embedding_dim']

        self.value_head = nn.Sequential(              # 预测每个城市对应的剩余长度V(s_i)
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # 输出每个城市的预测剩余路径长度
        )

    def forward(self, city_coordinates): # 这里的city_coordinates就是reset_state.problems
        # 编码所有城市
        city_embeddings = self.encoder(city_coordinates)  # shape: (batch, problem, embedding)
        
        # 预测每个城市对应的剩余长度V(s_i)
        predicted_lengths = self.value_head(city_embeddings).squeeze(-1)  # shape: (batch, problem)
        return predicted_lengths


# 3. 回溯策略网络
class BacktrackPolicyNetwork(nn.Module):
    def __init__(self, encoder: TSP_Encoder, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = encoder  # 共享编码器
        embedding_dim = self.model_params['embedding_dim']

        self.backtrack_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 64),  # 城市嵌入 + 全局上下文信息 + 价值网络预测
            nn.ReLU(),
            nn.Linear(64, 1)           # 输出该点的回溯潜力值
        )

    def forward(self, city_coordinates, current_tour, predicted_lengths_from_value_net): # current_tour: (batch, problem)
        # 获取城市嵌入
        city_embeddings = self.encoder(city_coordinates)  # shape: (batch, problem, embedding)
        
        # 获取每个城市作为回溯点的潜力值
        # 这里需要为每个batch中的每个城市计算潜力值
        batch_size, problem_size, embedding_dim = city_embeddings.shape

        # 计算全局上下文 (每个batch独立)
        global_context = city_embeddings.mean(dim=1)  # shape: (batch, embedding)
        
        # 扩展全局上下文，使其与city_embeddings的problem维度匹配
        global_context_expanded = global_context.unsqueeze(1).expand(batch_size, problem_size, embedding_dim)

        # 组合特征：每个城市嵌入、其对应的全局上下文以及价值网络预测值拼接
        combined_features = torch.cat([city_embeddings, global_context_expanded, predicted_lengths_from_value_net.unsqueeze(-1)], dim=-1) # shape: (batch, problem, embedding*2+1)
        
        # 计算回溯潜力值
        backtrack_potentials = self.backtrack_head(combined_features).squeeze(-1)  # shape: (batch, problem)
        
        # 掩码：只考虑当前路径中的城市作为回溯点
        # current_tour包含了路径中城市的索引，我们需要一个掩码来指示哪些城市在路径中
        # 假设current_tour中的值是0到problem_size-1的城市索引
        # 我们可以创建一个全False的掩码，然后将路径中的城市位置设为True
        
        # 实际上，backtrack_point的选择应该是在current_tour中的某个位置
        # 如果backtrack_point是路径中的一个索引，那么我们应该计算路径中每个位置的潜力
        # 这里的实现假设backtrack_point是城市ID，而不是路径中的位置
        # 如果是路径中的位置，则需要调整输入和计算方式
        
        # 暂时返回所有城市的潜力，后续根据实际需求添加掩码逻辑
        return backtrack_potentials


# 4. 重构策略网络
class ReconstructionPolicyNetwork(nn.Module):
    def __init__(self, encoder: TSP_Encoder, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = encoder  # 共享编码器
        embedding_dim = self.model_params['embedding_dim']

        # Head for selecting city_to_insert (选择要插入的城市)
        self.city_selection_head = nn.Sequential(
            nn.Linear(embedding_dim + 1, 64),  # 城市嵌入 + 价值网络预测
            nn.ReLU(),
            nn.Linear(64, 1) # Output a score for each city
        )

        # Head for selecting edge_to_insert (通过选择边的第一个节点来选择要插入的边)
        self.edge_selection_head = nn.Sequential(
            nn.Linear(embedding_dim + 1, 64),  # 城市嵌入 + 价值网络预测
            nn.ReLU(),
            nn.Linear(64, 1) # Output a score for each node (as start of an edge)
        )

    def forward(self, city_coordinates, current_tour, predicted_lengths_from_value_net):
        # city_coordinates: (batch, problem, 2)
        # current_tour: (batch, problem) - contains city indices in order
        # predicted_lengths_from_value_net: (batch, problem)

        city_embeddings = self.encoder(city_coordinates)  # shape: (batch, problem, embedding)

        # Predict probabilities for city_to_insert (预测要插入的城市的概率)
        # 拼接城市嵌入和价值网络预测
        city_features = torch.cat([city_embeddings, predicted_lengths_from_value_net.unsqueeze(-1)], dim=-1) # (batch, problem, embedding + 1)
        city_logits = self.city_selection_head(city_features).squeeze(-1) # shape: (batch, problem)
        city_to_insert_probs = F.softmax(city_logits, dim=-1)

        # Predict probabilities for edge_to_insert (预测要插入的边的概率)
        # We need embeddings of nodes in the current_tour to predict edges (我们需要当前路径中节点的嵌入来预测边)
        # For simplicity, let's use the city_embeddings directly and assume we are selecting a node (为简化起见，我们直接使用城市嵌入，并假设我们正在选择一个节点)
        # that *starts* an edge in the current tour. (该节点是当前路径中某条边的起始点。)
        # The actual edge will be (selected_node, next_node_in_tour) (实际的边将是 (selected_node, next_node_in_tour))

        # To get embeddings of nodes in current_tour: (获取当前路径中节点的嵌入:)
        batch_size, problem_size, embedding_dim = city_embeddings.shape
        gathering_index = current_tour.unsqueeze(2).expand(batch_size, problem_size, embedding_dim)
        tour_node_embeddings = city_embeddings.gather(dim=1, index=gathering_index) # shape: (batch, problem, embedding)
        
        # 拼接路径节点嵌入和价值网络预测
        gathering_index_value = current_tour.unsqueeze(2).expand(batch_size, problem_size, 1)
        tour_node_values = predicted_lengths_from_value_net.unsqueeze(-1).gather(dim=1, index=gathering_index_value) # shape: (batch, problem, 1)
        edge_features = torch.cat([tour_node_embeddings, tour_node_values], dim=-1) # (batch, problem, embedding + 1)

        edge_logits = self.edge_selection_head(edge_features).squeeze(-1) # shape: (batch, problem)
        edge_to_insert_probs = F.softmax(edge_logits, dim=-1)

        return city_to_insert_probs, edge_to_insert_probs


# 主GFlowNet TSP模型
class GFlowNetTSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        # 共享编码器
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)

        # 价值网络
        self.value_network = ValueNetwork(self.encoder, **model_params)

        # 回溯策略网络
        self.backtrack_policy_network = BacktrackPolicyNetwork(self.encoder, **model_params)

        # 重构策略网络（基于POMO解码器）
        self.reconstruction_policy_network = ReconstructionPolicyNetwork(self.encoder, **model_params)

        hidden_dim = model_params.get('backtrack_value_hidden_dim', 64)
        self.backtrack_value_head = nn.Sequential(
            nn.Linear(model_params['embedding_dim'], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    # TODO: 根据训练反馈调优回溯分数的估计模块
    def compute_backtrack_scores(
        self,
        problems: Tensor,
        tours: Tensor,
        node_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算批次中每条路径各位置的回溯分数 φ(i)，其中 φ(i)=前缀真实路程−预测剩余路程。

        Args:
            problems: 城市坐标张量，形状为 (batch, N, 2)。
            tours: 城市访问序列，形状为 (batch, N)。

        Returns:
            张量形状为 (batch, N)，为每个前缀位置的回溯分数。
        """
        # 获取节点嵌入并按路径顺序排列
        if node_embeddings is None:
            node_embeddings = self.encoder(problems)  # (batch, N, embedding)
        embedding_idx = tours.unsqueeze(-1).expand(-1, -1, node_embeddings.size(-1))
        tour_embeddings = node_embeddings.gather(dim=1, index=embedding_idx)

        # 预测剩余成本（不截断梯度以保持可微分）
        predicted_remaining = self.backtrack_value_head(tour_embeddings).squeeze(-1)

        # 计算路径前缀的真实旅行成本
        coord_idx = tours.unsqueeze(-1).expand(-1, -1, problems.size(-1))
        ordered_coords = problems.gather(dim=1, index=coord_idx)
        step_vectors = ordered_coords[:, 1:, :] - ordered_coords[:, :-1, :]
        step_cost = torch.norm(step_vectors, dim=-1)
        prefix_cost = torch.cat(
            [
                torch.zeros(step_cost.size(0), 1, device=problems.device, dtype=problems.dtype),
                torch.cumsum(step_cost, dim=1),
            ],
            dim=1,
        )

        return prefix_cost - predicted_remaining

    def reconstruct_from_prefix(
        self,
        problems: Tensor,
        prefixes: Sequence[Sequence[int]],
        num_samples: int,
        greedy: bool = False,
        node_embeddings: Optional[Tensor] = None,
        return_log_prob: bool = False,
    ) -> Tensor:
        """
        利用POMO解码器在给定路径前缀下生成多条完整候选路径。

        Args:
            problems: 城市坐标，形状 (batch, N, 2)。
            prefixes: 每个batch样本对应的前缀序列（至少1个节点）。
            num_samples: 为每个前缀生成的候选条数 m。
            greedy: True时使用贪心解码，否则按概率随机采样。
            node_embeddings: 可选的城市嵌入 (batch, N, embedding)，减少重复编码。

        Returns:
            Tensor: 形状 (batch, m, N) 的完整TSP路径集合。
        """
        if problems.dim() != 3:
            raise ValueError("problems 张量形状必须为 (batch, N, 2)")
        batch_size, problem_size, _ = problems.shape
        if len(prefixes) != batch_size:
            raise ValueError("prefixes 的长度必须等于 batch size")
        if num_samples <= 0:
            raise ValueError("num_samples 必须为正整数")

        device = problems.device
        if node_embeddings is None:
            node_embeddings = self.encoder(problems)
        embedding_dim = node_embeddings.size(-1)

        pomo_size = num_samples
        tours = torch.full(
            (batch_size, pomo_size, problem_size),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        visited_mask = torch.zeros(
            (batch_size, pomo_size, problem_size), dtype=torch.bool, device=device
        )
        steps_filled = torch.zeros(
            (batch_size, pomo_size), dtype=torch.long, device=device
        )
        current_node = torch.zeros(
            (batch_size, pomo_size), dtype=torch.long, device=device
        )
        first_nodes = torch.zeros(
            (batch_size, pomo_size), dtype=torch.long, device=device
        )

        for batch_idx, prefix in enumerate(prefixes):
            if len(prefix) == 0:
                raise ValueError("每个前缀至少包含1个城市")
            prefix_tensor = torch.as_tensor(prefix, dtype=torch.long, device=device)
            if prefix_tensor.numel() > problem_size:
                raise ValueError("前缀长度不能超过问题规模")

            seq_len = prefix_tensor.numel()
            tours[batch_idx, :, :seq_len] = prefix_tensor
            visited_mask[batch_idx, :, prefix_tensor] = True
            steps_filled[batch_idx, :] = seq_len
            current_node[batch_idx, :] = prefix_tensor[-1]
            first_nodes[batch_idx, :] = prefix_tensor[0]

        # 使用共享编码器+POMO解码器进行自回归采样
        self.decoder.set_kv(node_embeddings)
        batch_arange = torch.arange(batch_size, device=device).unsqueeze(1).expand(
            batch_size, pomo_size
        )
        first_embeddings = node_embeddings[batch_arange, first_nodes]
        self.decoder.set_q1(first_embeddings)

        tours_flat = tours.view(batch_size * pomo_size, problem_size)
        visited_flat = visited_mask.view(batch_size * pomo_size, problem_size)
        steps_flat = steps_filled.view(-1)
        current_flat = current_node.view(-1)
        if return_log_prob:
            log_prob_accum = torch.zeros(
                batch_size * pomo_size, dtype=problems.dtype, device=device
            )

        for _ in range(problem_size):
            # 自回归地对所有候选同时采样一个新城市，mask保证不会重复访问
            active = steps_flat < problem_size
            if not torch.any(active):
                break

            mask = visited_mask & active.view(batch_size, pomo_size, 1)
            ninf_mask = torch.zeros(
                (batch_size, pomo_size, problem_size),
                dtype=node_embeddings.dtype,
                device=device,
            )
            ninf_mask.masked_fill_(mask, float("-inf"))

            encoded_last = node_embeddings[batch_arange, current_node]
            probs = self.decoder(encoded_last, ninf_mask)
            probs_flat = probs.view(batch_size * pomo_size, problem_size)

            active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            if active_idx.numel() == 0:
                break

            if greedy:
                picked = probs_flat[active_idx].argmax(dim=-1)
            else:
                picked = torch.multinomial(probs_flat[active_idx], num_samples=1).squeeze(-1)

            if return_log_prob:
                chosen_prob = probs_flat[active_idx, picked].clamp_min(1e-9)
                log_prob_accum[active_idx] += torch.log(chosen_prob)

            tours_flat[active_idx, steps_flat[active_idx]] = picked
            visited_flat[active_idx, picked] = True
            current_flat[active_idx] = picked
            steps_flat[active_idx] += 1

        if (tours_flat == -1).any():
            raise RuntimeError("存在未完成的路径，检查前缀或采样逻辑是否正确")

        if return_log_prob:
            return tours, log_prob_accum.view(batch_size, pomo_size)
        return tours

    def sample_candidates(
        self,
        problems: Tensor,
        tours: Tensor,
        topk: int,
        num_reconstructions: int,
        greedy: bool = False,
    ) -> Tensor:
        """
        计算回溯分数，挑选Top-K前缀，并调用重构函数扩充候选路径。
        """
        if topk <= 0 or num_reconstructions <= 0:
            raise ValueError("topk 与 num_reconstructions 必须为正整数")

        scores = self.compute_backtrack_scores(problems, tours)
        k = min(topk, scores.size(1))
        _, indices = torch.topk(scores, k=k, dim=1)

        node_embeddings = self.encoder(problems)
        candidates = []
        for rank in range(k):
            prefix_ends = indices[:, rank]
            prefixes = [
                tours[batch_idx, : end_idx.item() + 1].tolist()
                for batch_idx, end_idx in enumerate(prefix_ends)
            ]
            rebuilt = self.reconstruct_from_prefix(
                problems,
                prefixes,
                num_reconstructions,
                greedy=greedy,
                node_embeddings=node_embeddings,
            )
            candidates.append(rebuilt.unsqueeze(1))

        return torch.cat(candidates, dim=1)

    def forward(self, city_coordinates, current_tour):
        # city_coordinates: (batch, problem, 2)
        # current_tour: (batch, problem) - current tour indices

        # 价值网络预测
        predicted_value = self.value_network(city_coordinates) # shape: (batch, 1)

        # 回溯策略网络计算回溯潜力
        backtrack_potentials = self.backtrack_policy_network(city_coordinates, current_tour, predicted_value) # shape: (batch, problem)

        # 重构策略网络计算城市和边的选择概率
        city_to_insert_probs, edge_to_insert_probs = self.reconstruction_policy_network(city_coordinates, current_tour, predicted_value)
        # city_to_insert_probs: (batch, problem)
        # edge_to_insert_probs: (batch, problem)

        return predicted_value, backtrack_potentials, city_to_insert_probs, edge_to_insert_probs