import torch
import torch.nn as nn
import torch.nn.functional as F

from GFlowNetTSPModel import GFlowNetTSPModel
from ImpTSPEnv import TSPEnv_Improve, ImproveState

class GFlowNetAgent(nn.Module):
    def __init__(self, env: TSPEnv_Improve, model: GFlowNetTSPModel, **agent_params):
        super().__init__()
        self.env = env
        self.model = model
        self.agent_params = agent_params

    def select_action(self, state: ImproveState):
        # 获取当前路径和城市坐标
        current_tour = state.current_tour
        city_coordinates = self.env.problems # (batch, problem, 2)

        batch_size = current_tour.shape[0]
        problem_size = current_tour.shape[1]

        # 通过模型获取策略输出
        _, backtrack_potentials, city_to_insert_probs, edge_to_insert_probs = self.model(city_coordinates, current_tour)

        # 1. 选择backtrack_point
        # 假设backtrack_potentials是logits，进行softmax转换为概率
        backtrack_probs = F.softmax(backtrack_potentials, dim=-1)
        backtrack_point = torch.multinomial(backtrack_probs, 1).squeeze(1) # (batch,)

        # 2. 选择city_to_insert
        city_to_insert = torch.multinomial(city_to_insert_probs, 1).squeeze(1) # (batch,)

        # 3. 选择edge_to_insert
        # edge_to_insert_probs是选择边的起点的概率
        # 掩码处理：确保edge_start_node_idx不与city_to_insert相同
        edge_selection_mask = torch.ones_like(edge_to_insert_probs, dtype=torch.bool)
        edge_selection_mask[torch.arange(batch_size), city_to_insert] = False
        masked_edge_to_insert_probs = edge_to_insert_probs.masked_fill(~edge_selection_mask, 1e-9) # 将被掩码的概率设为极小值
        masked_edge_to_insert_probs = masked_edge_to_insert_probs / masked_edge_to_insert_probs.sum(dim=-1, keepdim=True) # 重新归一化

        edge_start_node_idx = torch.multinomial(masked_edge_to_insert_probs, 1).squeeze(1) # (batch,)

        # 根据edge_start_node_idx和current_tour确定完整的edge_to_insert (start_node, end_node)
        batch_size = current_tour.shape[0]
        problem_size = current_tour.shape[1]

        # 找到edge_start_node_idx在current_tour中的位置
        # current_tour: (batch, problem)
        # edge_start_node_idx: (batch,)
        
        # 创建一个与current_tour相同形状的张量，用于比较
        expanded_edge_start_node_idx = edge_start_node_idx.unsqueeze(1).expand_as(current_tour)
        
        # 找到匹配的位置
        match_indices = (current_tour == expanded_edge_start_node_idx).nonzero(as_tuple=True)
        
        # match_indices[0] 是batch索引，match_indices[1] 是在current_tour中的位置
        # 确保每个batch只有一个匹配
        assert torch.all(torch.bincount(match_indices[0]) == 1), "每个batch应该只有一个匹配的起点"

        # 获取每个batch中edge_start_node_idx在current_tour中的位置
        positions_in_tour = match_indices[1] # (batch,)

        # 计算下一个节点的索引 (循环)
        next_positions_in_tour = (positions_in_tour + 1) % problem_size

        # 获取edge_start_node和edge_end_node
        edge_start_node = current_tour[torch.arange(batch_size), positions_in_tour]
        edge_end_node = current_tour[torch.arange(batch_size), next_positions_in_tour]

        edge_to_insert = torch.stack([edge_start_node, edge_end_node], dim=1) # (batch, 2)

        return backtrack_point, city_to_insert, edge_to_insert

    def run_episode(self):
        # 重置环境
        state = self.env.reset()
        
        episode_rewards = []
        episode_states = [state]
        
        while not state.done:
            # 代理选择动作
            backtrack_point, city_to_insert, edge_to_insert = self.select_action(state)
            
            # 环境执行动作
            next_state, reward, done = self.env.step(state, backtrack_point, city_to_insert, edge_to_insert)
            
            episode_rewards.append(reward)
            episode_states.append(next_state)
            state = next_state
            
        return episode_states, episode_rewards