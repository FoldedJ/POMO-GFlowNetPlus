import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from ImpTSPEnv import TSPEnv_Improve
from GFlowNetTSPModel import GFlowNetTSPModel
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold


class GFlowNetTBTrainer:
    """
    GFlowNet 轨迹平衡训练器基类，仅封装核心训练逻辑，配置由外部脚本注入。
    """

    def __init__(
        self,
        batch_size: int,
        problem_size: int,
        max_steps: int | None,
        embedding_dim: int,
        head_num: int,
        lr: float,
        epochs: int,
        eval_interval: int,
        save_interval: int,
        save_dir: str,
        encoder_layer_num: int,
        ff_hidden_dim: int,
        log_dir: str,
        scaling_factor: float,
        grad_clip: float,
        temp_start: float,
        temp_min: float,
        temp_decay: float,
        patience: int,
        candidate_topk: int = 3,
        samples_per_prefix: int = 2,
    ) -> None:
        if max_steps is None:
            max_steps = problem_size * 2

        self.batch_size = batch_size
        self.problem_size = problem_size
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.scaling_factor = scaling_factor
        self.grad_clip = grad_clip
        self.temp_start = temp_start
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.patience = patience
        self.candidate_topk = candidate_topk
        self.samples_per_prefix = samples_per_prefix

        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "train.log")),
                logging.StreamHandler(),
            ],
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)

        model_params = {
            "embedding_dim": embedding_dim,
            "head_num": head_num,
            "qkv_dim": max(1, embedding_dim // head_num),
            "encoder_layer_num": encoder_layer_num,
            "ff_hidden_dim": ff_hidden_dim,
            "sqrt_embedding_dim": embedding_dim ** 0.5,
            "problem_size": problem_size,
        }

        self.env = TSPEnv_Improve(
            device=self.device,
            problem_size=problem_size,
            max_steps=max_steps,
            patience=patience,
        )
        self.model = GFlowNetTSPModel(**model_params).to(self.device)
        self.log_z = nn.Parameter(torch.zeros(1, device=self.device))
        self.trainable_params = list(self.model.parameters()) + [self.log_z]
        self.optimizer = optim.Adam(self.trainable_params, lr=lr)

    def _temperature(self, epoch: int) -> float:
        return max(self.temp_min, self.temp_start * (self.temp_decay ** epoch))

    def train_loop(self) -> None:
        self.model.train()
        os.makedirs(self.save_dir, exist_ok=True)

        for epoch in range(self.epochs):
            problems = get_random_problems(self.batch_size, self.problem_size).to(self.device)
            problems = augment_xy_data_by_8_fold(problems)
            temperature = self._temperature(epoch)

            # TB workflow: episode -> optimizer.optimize_episode() -> collect trajectory -> compute log_pf/log_pb -> TB loss -> backward -> optimizer step
            state = self.env.reset(problems)
            trajectory = self.env.optimize_episode(
                model=self.model,
                current_tour=state.current_tour,
                topk=self.candidate_topk,
                num_samples=self.samples_per_prefix,
                greedy=False,
            )
            log_pf = trajectory.log_pf
            log_pb = trajectory.log_pb
            reward = trajectory.reward

            loss = (self.log_z + log_pf - log_pb - reward).pow(2).mean()
            if torch.isnan(loss):
                logging.warning("发现NaN损失，跳过当前batch。")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.trainable_params, self.grad_clip)
            self.optimizer.step()

            logging.info(
                "Epoch %d/%d | Loss %.4f | log_pf %.4f | log_pb %.4f | reward %.4f | temp %.3f",
                epoch + 1,
                self.epochs,
                loss.item(),
                log_pf.mean().item(),
                log_pb.mean().item(),
                reward.mean().item(),
                temperature,
            )

            if (epoch + 1) % self.eval_interval == 0:
                eval_length = evaluate(self.model, self.problem_size, self.device)
                logging.info("Eval avg tour length: %.4f", eval_length)

            if (epoch + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f"gflownet_tsp_epoch_{epoch+1}.pt")
                save_model(self.model, self.optimizer, self.log_z, epoch, checkpoint_path)

        logging.info("训练完成！")

        # 旧版POMO强化学习训练逻辑（保留以备参考）
        """
        legacy_state = self.env.reset(problems)
        # 旧逻辑：调用GFlowNetAgent.run_episode收集轨迹后，分别计算value/policy consistency等损失。
        # 该段代码已被轨迹平衡训练替换，如需回退可在此处接入旧的训练流程。
        """


def evaluate(model: GFlowNetTSPModel, problem_size: int, device: torch.device, num_problems: int = 100) -> float:
    model.eval()
    env = TSPEnv_Improve(device=device, problem_size=problem_size, max_steps=problem_size * 2)

    with torch.no_grad():
        problems = get_random_problems(num_problems, problem_size).to(device)
        state = env.reset(problems)
        current_tours = state.current_tour
        node_embeddings = model.encoder(problems)
        scores = model.compute_backtrack_scores(problems, current_tours, node_embeddings=node_embeddings)
        prefix_index = scores.argmax(dim=-1)
        prefixes = [
            current_tours[b, : prefix_index[b].item() + 1].tolist()
            for b in range(current_tours.size(0))
        ]
        rebuilt = model.reconstruct_from_prefix(
            problems,
            prefixes,
            num_samples=1,
            greedy=True,
            node_embeddings=node_embeddings,
            return_log_prob=False,
        )
        rebuilt = rebuilt.squeeze(1)
        lengths = env.compute_tour_lengths(rebuilt)
        avg_length = lengths.mean().item()

    model.train()
    return avg_length


def save_model(model, optimizer, log_z, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "log_z": log_z.detach().cpu(),
        },
        path,
    )
    logging.info("模型已保存到 %s (Epoch %d)", path, epoch)


def load_model(model, optimizer, log_z, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "log_z" in checkpoint:
        log_z.data.copy_(checkpoint["log_z"].to(device))
    epoch = checkpoint["epoch"]
    logging.info("模型已从 %s 加载 (Epoch %d)", path, epoch)
    return epoch
