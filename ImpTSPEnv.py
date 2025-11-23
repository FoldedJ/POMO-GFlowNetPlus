from dataclasses import dataclass
import torch

from TSProblemDef import get_random_problems


@dataclass
class ImproveState:
    problems: torch.Tensor          
    # (batch, problem, 2)
    current_tour: torch.Tensor # 当前完整路径      
    # (batch, problem)
    path_length: torch.Tensor       
    # (batch,)
    done: bool # 是否结束？


class TSPEnv_Improve:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.batch_size = None
        self.problems = None        
        # (batch, problem, 2)

    ###################################################################
    # 初始化问题数据
    ###################################################################
    def load_problems(self, batch_size):
        self.batch_size = batch_size
        self.problems = get_random_problems(batch_size, self.problem_size)

    ###################################################################
    # 环境RESET：生成初始解
    ###################################################################
    def reset(self, initial_method="random"):
        # 随机生成一条路径
        if initial_method == "random":
            initial_tour = torch.stack([
                torch.randperm(self.problem_size) for _ in range(self.batch_size)
            ])
        else:
            raise NotImplementedError

        path_length = self._get_tour_length(initial_tour)

        state = ImproveState(
            problems=self.problems,
            current_tour=initial_tour,
            path_length=path_length,
            done=False
        )

        return state

    ###################################################################
    # 状态转移
    ###################################################################
    def update(self, new_tour):
        new_length = self._get_tour_length(new_tour)

        state = ImproveState(
            problems=self.problems,
            current_tour=new_tour,
            path_length=new_length,
            done=False
        )

        return state

    ###################################################################
    # 终止条件检测
    ###################################################################
    def finish(self, improved):
        done = (improved.sum() == 0)
        return done

    ###################################################################
    # 计算路径长度
    ###################################################################
    def _get_tour_length(self, tour):
        batch = self.batch_size
        problem = self.problem_size

        # 获取路径节点坐标
        idx = tour.unsqueeze(2).expand(batch, problem, 2)
        ordered_seq = self.problems.gather(dim=1, index=idx)

        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()

        return segment_lengths.sum(1)
