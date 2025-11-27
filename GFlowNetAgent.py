import torch
import torch.nn as nn
import torch.nn.functional as F

from GFlowNetTSPModel import GFlowNetTSPModel
from ImpTSPEnv import TSPEnv_Improve, ImproveState

class GFlowNetAgent(nn.Module):
    """GFlowNet代理，用于在TSP改进环境中选择动作并运行episode。"""
    def __init__(self, env: TSPEnv_Improve, model: GFlowNetTSPModel, **agent_params):
        super().__init__()
        self.env = env  # TSP改进环境实例
        self.model = model  # GFlowNetTSPModel实例
        self.agent_params = agent_params  # 代理参数

    def select_action(self, state: ImproveState):
        # 获取当前路径和城市坐标
        current_tour = state.current_tour
        city_coordinates = self.env.problems # (batch, problem, 2)

        batch_size = current_tour.shape[0]
        problem_size = current_tour.shape[1]

        # 通过模型获取策略输出
        predicted_value, backtrack_potentials, city_to_insert_probs, edge_to_insert_probs = self.model(city_coordinates, current_tour)

        # 1. 选择回溯点 (backtrack_point)
        # 假设backtrack_potentials是logits，进行softmax转换为概率
        backtrack_log_probs = F.log_softmax(backtrack_potentials, dim=-1)
        backtrack_probs = F.softmax(backtrack_potentials, dim=-1)
        
        num_candidates = self.agent_params.get('num_candidates', 3) # 从agent_params获取num_candidates，默认为3
        
        # 从概率分布中采样num_candidates个回溯点
        backtrack_candidates = torch.multinomial(backtrack_probs, num_candidates, replacement=True) # (batch, num_candidates)
        # 收集采样到的回溯点的对数概率
        backtrack_action_log_probs = backtrack_log_probs.gather(dim=-1, index=backtrack_candidates) # (batch, num_candidates)

        # 2. 选择要插入的城市 (city_to_insert)
        city_to_insert_log_probs = F.log_softmax(city_to_insert_probs, dim=-1)
        # 从概率分布中采样num_candidates个要插入的城市
        city_to_insert_candidates = torch.multinomial(city_to_insert_probs, num_candidates, replacement=True) # (batch, num_candidates)
        # 收集采样到的要插入的城市的对数概率
        city_to_insert_action_log_probs = city_to_insert_log_probs.gather(dim=-1, index=city_to_insert_candidates) # (batch, num_candidates)

        # 3. 选择要插入的边 (edge_to_insert)
        # edge_to_insert_probs是选择边的起点的概率
        # 掩码处理：确保edge_start_node_idx不与city_to_insert_candidates相同
        # edge_selection_mask: (batch, num_candidates, problem_size)
        edge_selection_mask = torch.ones(batch_size, num_candidates, problem_size, dtype=torch.bool, device=city_coordinates.device)
        
        # 将每个batch中每个候选的city_to_insert位置设为False，因为不能将城市插入到自身
        # city_to_insert_candidates: (batch, num_candidates)
        # 扩展city_to_insert_candidates以匹配problem_size维度
        # city_to_insert_candidates_expanded: (batch, num_candidates, 1)
        city_to_insert_candidates_expanded = city_to_insert_candidates.unsqueeze(-1)
        
        # 使用scatter_将对应位置设为False
        edge_selection_mask.scatter_(-1, city_to_insert_candidates_expanded, False)
        
        # 扩展edge_to_insert_probs以匹配num_candidates维度
        # edge_to_insert_probs_expanded: (batch, 1, problem_size)
        edge_to_insert_probs_expanded = edge_to_insert_probs.unsqueeze(1)
        
        # 应用掩码，将不能选择的边的概率设为极小值，避免采样到
        masked_edge_to_insert_probs = edge_to_insert_probs_expanded.masked_fill(~edge_selection_mask, 1e-9) # 将被掩码的概率设为极小值
        # 重新归一化概率分布
        masked_edge_to_insert_probs = masked_edge_to_insert_probs / masked_edge_to_insert_probs.sum(dim=-1, keepdim=True) # 重新归一化
        
        masked_edge_to_insert_log_probs = torch.log(masked_edge_to_insert_probs)

        # 为torch.multinomial重塑张量: (batch_size * num_candidates, problem_size)
        reshaped_probs = masked_edge_to_insert_probs.view(-1, problem_size)
        
        # 从重塑后的概率分布中采样
        # edge_start_node_candidates_flat: (batch_size * num_candidates, 1)
        edge_start_node_candidates_flat = torch.multinomial(reshaped_probs, 1, replacement=True)
        
        # 重塑回 (batch_size, num_candidates)
        edge_start_node_candidates = edge_start_node_candidates_flat.view(batch_size, num_candidates)
        
        # 收集采样到的边的起点的对数概率，调整索引以匹配原始3D形状
        edge_to_insert_action_log_probs = masked_edge_to_insert_log_probs.gather(dim=-1, index=edge_start_node_candidates.unsqueeze(-1)).squeeze(-1) # (batch, num_candidates)

        # 根据edge_start_node_candidates和current_tour确定完整的edge_to_insert (start_node, end_node)
        # edge_start_node_candidates: (batch, num_candidates)
        # current_tour: (batch, problem_size)

        # 扩展current_tour以匹配num_candidates维度
        # current_tour_expanded: (batch, num_candidates, problem_size)
        current_tour_expanded = current_tour.unsqueeze(1).expand(batch_size, num_candidates, problem_size)

        # 扩展edge_start_node_candidates以匹配problem_size维度
        # edge_start_node_candidates_expanded: (batch, num_candidates, 1)
        edge_start_node_candidates_expanded = edge_start_node_candidates.unsqueeze(2)

        # 找到每个候选起点在current_tour中的位置
        # match_mask: (batch, num_candidates, problem_size)
        match_mask = (current_tour_expanded == edge_start_node_candidates_expanded)

        # positions_in_tour: (batch, num_candidates)
        positions_in_tour = torch.argmax(match_mask.int(), dim=-1)

        # 计算下一个节点的索引 (循环，因为是环形路径)
        next_positions_in_tour = (positions_in_tour + 1) % problem_size

        # 获取边的起始节点和结束节点
        # edge_start_node: (batch, num_candidates)
        edge_start_node = current_tour_expanded.gather(dim=-1, index=positions_in_tour.unsqueeze(-1)).squeeze(-1)
        # edge_end_node: (batch, num_candidates)
        edge_end_node = current_tour_expanded.gather(dim=-1, index=next_positions_in_tour.unsqueeze(-1)).squeeze(-1)

        # edge_to_insert: (batch, num_candidates, 2) 包含起始节点和结束节点
        edge_to_insert_candidates = torch.stack([edge_start_node, edge_end_node], dim=-1)

        # 将三个动作的对数概率相加，得到总的动作对数概率
        action_log_probs = backtrack_action_log_probs + city_to_insert_action_log_probs + edge_to_insert_action_log_probs # (batch, num_candidates)

        # 返回所有候选动作、它们的对数概率以及预测值
        return backtrack_candidates, city_to_insert_candidates, edge_to_insert_candidates, action_log_probs, predicted_value.repeat(1, num_candidates)

    def run_episode(self, problems):
        # 重置环境，开始新的episode
        state = self.env.reset(problems)
        
        episode_rewards = [] # 存储每个时间步的奖励
        episode_states = [state] # 存储每个时间步的状态
        episode_action_log_probs = [] # 存储每个时间步的动作对数概率
        episode_value_preds = [] # 存储每个时间步的价值预测
        
        # 循环直到所有环境中的路径都完成改进
        while not state.done.all():
            # 代理选择3个动作候选
            backtrack_candidates, city_candidates, edge_candidates, action_log_probs, value_candidates = self.select_action(state)
            batch_size = backtrack_candidates.shape[0]
            num_candidates = 3

            # 执行每个候选动作获取下一个状态和奖励
            next_states = []
            rewards = []
            dones = []
            for i in range(num_candidates):
                bt = backtrack_candidates[:, i]
                ct = city_candidates[:, i]
                et = edge_candidates[:, i, :]
                # 在环境中执行动作，获取下一个状态、奖励和完成标志
                ns, r, d = self.env.step(state, bt, ct, et)
                next_states.append(ns)
                rewards.append(r)
                dones.append(d)

            # 转换为张量以便选择最佳候选
            rewards_tensor = torch.stack(rewards, dim=1) # (batch, 3)
            # 选择每个batch中奖励最高的候选索引
            best_candidate_indices = torch.argmax(rewards_tensor, dim=1) # (batch,)

            # 收集最佳候选的下一个状态信息
            # 从所有候选的next_states中，根据best_candidate_indices选择最佳的下一个状态
            best_current_tour = torch.stack([ns.current_tour for ns in next_states], dim=1)[torch.arange(batch_size), best_candidate_indices]
            best_step_count = torch.stack([ns.step_count for ns in next_states], dim=1)[torch.arange(batch_size), best_candidate_indices]
            best_path_length = torch.stack([ns.path_length for ns in next_states], dim=1)[torch.arange(batch_size), best_candidate_indices]
            best_done = torch.stack([ns.done for ns in next_states], dim=1)[torch.arange(batch_size), best_candidate_indices]
            # 构建最佳的下一个状态对象
            best_next_state = ImproveState(current_tour=best_current_tour, step_count=best_step_count, path_length=best_path_length, done=best_done)

            # 收集最佳候选的奖励、动作对数概率和价值预测
            best_reward = rewards_tensor[torch.arange(batch_size), best_candidate_indices]
            best_action_log_prob = action_log_probs[torch.arange(batch_size), best_candidate_indices]
            best_value_pred = value_candidates[torch.arange(batch_size), best_candidate_indices]

            # 将最佳结果加入episode列表
            episode_rewards.append(best_reward)
            episode_states.append(best_next_state)
            episode_action_log_probs.append(best_action_log_prob)
            episode_value_preds.append(best_value_pred)
            # 更新当前状态为最佳的下一个状态，继续循环
            state = best_next_state
            
        # 返回episode中收集到的所有状态、奖励、动作对数概率和价值预测
        return episode_states, episode_rewards, episode_action_log_probs, episode_value_preds