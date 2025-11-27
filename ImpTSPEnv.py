from dataclasses import dataclass
import torch

from TSProblemDef import get_random_problems


@dataclass
class ImproveState:
    current_tour: torch.Tensor  # 当前完整路径，shape: (batch, problem)
    path_length: torch.Tensor   # 当前路径长度
    step_count: torch.Tensor             # 当前已执行的步数 ？ 为什么要记录步数###############################
    done: torch.Tensor                  # 标志，表示当前episode是否结束


class TSPEnv_Improve:
    def __init__(self, device, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']  # TSP问题的规模（城市数量）
        # 最大步数，用于限制每个episode的迭代次数，默认为问题规模的两倍 ? 步数到底是干什么###################################
        self.max_steps = env_params.get('max_steps', self.problem_size * 2)
        self.eta = env_params.get('eta', 1.0) # GFlowNet奖励函数的eta参数
        self.batch_size = None  
        self.problems = None    # shape: (batch, problem, 2)
        self.device = device

    def __call__(self, problems):
        self.batch_size = problems.shape[0]
        self.problems = problems

    ###################################################################
    # 初始化问题数据
    ###################################################################
    def load_problems(self, batch_size):
        """
        加载TSP问题实例。
        """
        self.batch_size = batch_size
        self.problems = get_random_problems(batch_size, self.problem_size)

    ###################################################################
    # 环境RESET：生成初始解
    ###################################################################
    def reset(self, problems, initial_method="random"):
        """
        重置环境，生成一个初始的TSP路径。
        Args:
            initial_method (str): 初始路径的生成方法，目前只支持 "random"。
        Raises:
            NotImplementedError: 如果使用了不支持的初始化方法。
        """
        self(problems) # Store problems and batch_size
        # 随机生成一条路径
        if initial_method == "random":
            initial_tour = torch.stack([
                torch.randperm(self.problem_size, device=problems.device) for _ in range(self.batch_size)
            ])
        else:
            raise NotImplementedError

        path_length = self._get_tour_length(initial_tour)

        state = ImproveState(
            current_tour=initial_tour,
            path_length=path_length,
            step_count=torch.zeros(self.batch_size, dtype=torch.long, device=self.device),
            done=torch.full((self.batch_size,), False, device=problems.device, dtype=torch.bool)
        )

        return state

    def step(self, state: ImproveState, backtrack_point, city_to_insert, edge_to_insert):
        old_tour = state.current_tour
        old_length = state.path_length
        old_step_count = state.step_count

        # 1. 重构路径
        new_tour = self._reconstruct_tour(old_tour, backtrack_point, city_to_insert, edge_to_insert)
        new_length = self._get_tour_length(new_tour)

        # 2. 计算奖励
        # 奖励为路径质量奖励：R(pi) = exp(-eta * L(pi))
        reward = torch.exp(-self.eta * new_length)

        # 3. 更新步数
        new_step_count = old_step_count + 1

        # 4. 检查终止条件
        # 当达到最大步数时，剧集结束
        done = (new_step_count >= self.max_steps)

        next_state = ImproveState(
            current_tour=new_tour,
            path_length=new_length,
            step_count=new_step_count,
            done=done
        )

        return next_state, reward, done

    ###################################################################
    # 计算路径长度
    ###################################################################
    def _get_tour_length(self, tour):
        """
        计算给定TSP路径的总长度。
        Args:
            tour (torch.Tensor): TSP路径，shape: (batch, problem)。
        Returns:
            torch.Tensor: 路径的总长度，shape: (batch,)。
        """
        batch = self.batch_size
        problem = self.problem_size

        # 获取路径节点坐标
        # 将路径中的城市索引转换为实际的城市坐标
        idx = tour.unsqueeze(2).expand(batch, problem, 2)
        ordered_seq = self.problems.gather(dim=1, index=idx)

        # 将路径首尾相连，形成一个环
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        # 计算每段路径的长度 (欧几里得距离)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()

        # 返回所有段长度的总和
        return segment_lengths.sum(1)

    def _reconstruct_tour(self, current_tour, backtrack_point, city_to_insert, edge_to_insert):
        """
        根据回溯点、待插入城市和插入边重构TSP路径。
        Args:
            current_tour (torch.Tensor): 当前的TSP路径，shape: (batch, problem)。
            backtrack_point (torch.Tensor): 回溯点的索引，形状: (batch,)
            city_to_insert (torch.Tensor): 待插入城市的索引，形状: (batch,)
            edge_to_insert (torch.Tensor): 插入边的两个节点索引，形状: (batch, 2)
        Returns:
            torch.Tensor: 重构后的新TSP路径，shape: (batch, problem)。
        """
        # current_tour: (batch, problem)
        # backtrack_point: (batch,) - 回溯点的索引
        # city_to_insert: (batch,) - 待插入城市的索引
        # edge_to_insert: (batch, 2) - 构成插入边的两个节点索引

        batch_size = current_tour.shape[0]
        problem_size = current_tour.shape[1]

        new_tour = current_tour.clone()

        # 1. 从当前路径中移除待插入城市（矢量化实现，无batch循环）
        # 获取每个batch中待插入城市的索引
        city_eq_mask = (current_tour == city_to_insert.unsqueeze(1))  # shape: (batch, problem)
        remove_indices = torch.argmax(city_eq_mask.int(), dim=1)     # shape: (batch,)
        # 创建掩码排除待移除城市的索引
        batch_range = torch.arange(batch_size, device=current_tour.device)
        problem_range = torch.arange(problem_size, device=current_tour.device)
        keep_mask = (problem_range != remove_indices.unsqueeze(1))   # shape: (batch, problem)
        # 生成移除城市后的路径
        tour_without_city = current_tour[keep_mask].view(batch_size, problem_size - 1)  # shape: (batch, problem-1)

        # 2. 找到插入位置：插入边edge_to_insert的第一个节点之后（矢量化实现）
        edge_eq_mask = (tour_without_city == edge_to_insert[:, 0].unsqueeze(1))  # shape: (batch, problem-1)
        insert_positions = torch.argmax(edge_eq_mask.int(), dim=1)               # shape: (batch,)

        new_tours_list = []

        for b in range(batch_size):
            # 遍历批次中的每个样本
            current_tour_b = current_tour[b] # (problem,)
            city_to_insert_b = city_to_insert[b] # 标量
            edge_to_insert_b = edge_to_insert[b] # (2,)

            # 1. 从 current_tour_b 中移除待插入城市 city_to_insert_b
            # 找到 city_to_insert_b 在 current_tour_b 中的索引
            remove_idx_b = (current_tour_b == city_to_insert_b).nonzero(as_tuple=True)[0].item()

            # 创建移除城市后的路径 tour_without_city_b
            tour_without_city_b = torch.cat([
                current_tour_b[:remove_idx_b],
                current_tour_b[remove_idx_b+1:]
            ]) # (problem - 1,)

            # 2. 找到插入位置
            # 找到 edge_to_insert_b[0] 在 tour_without_city_b 中的索引
            # 插入操作发生在该节点之后
            insert_node_idx_in_tour_without_city_b = (tour_without_city_b == edge_to_insert_b[0]).nonzero(as_tuple=True)[0].item()
            insert_position_b = insert_node_idx_in_tour_without_city_b + 1 # 在该节点之后插入

            # 3. 将 city_to_insert_b 插入到 tour_without_city_b 中
            new_tour_b = torch.cat([
                tour_without_city_b[:insert_position_b],
                city_to_insert_b.unsqueeze(0), # 将其转换为一个单元素张量以便拼接
                tour_without_city_b[insert_position_b:]
            ]) # (problem,)
            new_tours_list.append(new_tour_b)

        new_tour = torch.stack(new_tours_list, dim=0) # (batch, problem)
        return new_tour
