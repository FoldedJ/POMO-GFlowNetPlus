from dataclasses import dataclass
import logging
import torch

from TSProblemDef import get_random_problems

logger = logging.getLogger(__name__)


@dataclass
class ImproveState:
    current_tour: torch.Tensor  # 当前完整路径，shape: (batch, problem)
    path_length: torch.Tensor   # 最佳路径长度
    step_count: torch.Tensor    # 当前已执行的改进次数
    done: torch.Tensor          # 标志，表示每个实例是否结束
    best_tour: torch.Tensor
    best_length: torch.Tensor
    no_improve: torch.Tensor


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
        self.patience = env_params.get('patience', 5)

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

        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        state = ImproveState(
            current_tour=initial_tour,
            path_length=path_length.clone(),
            step_count=zeros.clone(),
            done=torch.full((self.batch_size,), False, device=problems.device, dtype=torch.bool),
            best_tour=initial_tour.clone(),
            best_length=path_length.clone(),
            no_improve=zeros.clone()
        )

        return state

    def step(self, state: ImproveState, backtrack_point, city_to_insert, edge_to_insert):
        old_tour = state.current_tour
        old_step_count = state.step_count
        active_mask = ~state.done

        safe_bt = backtrack_point.clone()
        safe_ct = city_to_insert.clone()
        safe_edge = edge_to_insert.clone()
        inactive_idx = torch.nonzero(~active_mask, as_tuple=False).flatten()
        if inactive_idx.numel() > 0:
            safe_bt[inactive_idx] = state.best_tour[inactive_idx, 0]
            safe_ct[inactive_idx] = state.best_tour[inactive_idx, 0]
            safe_edge[inactive_idx] = state.best_tour[inactive_idx][:, :2]

        if active_mask.any():
            rebuilt = self._reconstruct_tour(old_tour, safe_bt, safe_ct, safe_edge)
        else:
            rebuilt = state.best_tour.clone()

        new_tour = torch.where(active_mask.unsqueeze(1), rebuilt, state.best_tour)
        new_length = self._get_tour_length(new_tour)
        candidate_length = torch.where(active_mask, new_length, state.best_length)
        better_mask = candidate_length < state.best_length
        updated_best_tour = torch.where(better_mask.unsqueeze(1), new_tour, state.best_tour)
        updated_best_length = torch.where(better_mask, candidate_length, state.best_length)

        no_improve = state.no_improve.clone()
        improve_idx = active_mask & better_mask
        stagnate_idx = active_mask & (~better_mask)
        no_improve[improve_idx] = 0
        no_improve[stagnate_idx] += 1

        reward = torch.exp(-self.eta * updated_best_length)
        new_step_count = torch.where(active_mask, old_step_count + 1, old_step_count)

        patience_hit = no_improve >= self.patience
        done = patience_hit
        newly_finished = (~state.done) & done
        if newly_finished.any():
            for b in torch.nonzero(newly_finished, as_tuple=False).flatten():
                steps = new_step_count[b].item()
                logger.debug(f"[STOP] batch={b.item()}, converged after {steps} improvement steps.")

        next_state = ImproveState(
            current_tour=updated_best_tour.clone(),
            path_length=updated_best_length.clone(),
            step_count=new_step_count,
            done=done,
            best_tour=updated_best_tour.clone(),
            best_length=updated_best_length.clone(),
            no_improve=no_improve
        )

        return next_state, reward, done

    ###################################################################
    # 计算路径长度
    ###################################################################
    def compute_tour_lengths(self, tours: torch.Tensor) -> torch.Tensor:
        """
        计算单条或多条候选路径的总长度，自动适配 (batch, N) 或 (batch, C, N)。
        Args:
            tours: 路径索引张量，最后一维长度等于问题规模。
        Returns:
            torch.Tensor: 若输入为 (batch, N) 则输出 (batch,)，否则输出 (batch, C)。
        """
        if self.problems is None:
            raise ValueError("请先调用 reset() 或显式设置 self.problems。")
        if tours.dim() not in (2, 3):
            raise ValueError("tours 的形状必须为 (batch, N) 或 (batch, C, N)。")

        single_tour = tours.dim() == 2
        if single_tour:
            tours = tours.unsqueeze(1)  # (batch, 1, N)

        batch_size, candidate_size, seq_len = tours.shape
        if seq_len != self.problem_size:
            raise ValueError("路径长度必须与 problem_size 一致。")

        problems = self.problems.unsqueeze(1).expand(batch_size, candidate_size, seq_len, 2)
        gather_idx = tours.unsqueeze(-1).expand(-1, -1, -1, 2)
        ordered_seq = problems.gather(dim=2, index=gather_idx)

        # 将路径首尾相连，批量计算每段欧几里得距离
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = torch.norm(ordered_seq - rolled_seq, dim=-1)
        tour_lengths = segment_lengths.sum(dim=2)

        return tour_lengths.squeeze(1) if single_tour else tour_lengths

    def _get_tour_length(self, tour: torch.Tensor) -> torch.Tensor:
        """
        兼容旧接口，内部调用 compute_tour_lengths。
        """
        return self.compute_tour_lengths(tour)

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
