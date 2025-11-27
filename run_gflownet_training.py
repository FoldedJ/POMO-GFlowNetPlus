import argparse
from dataclasses import dataclass, asdict

from train_gflownet import GFlowNetTBTrainer


@dataclass
class TrainerConfig:
    batch_size: int = 16
    problem_size: int = 20
    max_steps: int | None = None
    embedding_dim: int = 128
    head_num: int = 8
    lr: float = 1e-4
    epochs: int = 100
    eval_interval: int = 10
    save_interval: int = 10
    save_dir: str = "./checkpoints"
    encoder_layer_num: int = 1
    ff_hidden_dim: int = 512
    log_dir: str = "./logs"
    scaling_factor: float = 0.01
    grad_clip: float = 1.0
    temp_start: float = 1.2
    temp_min: float = 0.1
    temp_decay: float = 0.995
    patience: int = 5


def parse_args() -> TrainerConfig:
    parser = argparse.ArgumentParser(description="Run GFlowNet Trajectory Balance training.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--problem_size", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--head_num", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--encoder_layer_num", type=int, default=1)
    parser.add_argument("--ff_hidden_dim", type=int, default=512)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--scaling_factor", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--temp_start", type=float, default=1.2)
    parser.add_argument("--temp_min", type=float, default=0.1)
    parser.add_argument("--temp_decay", type=float, default=0.995)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    return TrainerConfig(**vars(args))


def main():
    config = parse_args()
    trainer = GFlowNetTBTrainer(**asdict(config))
    trainer.train_loop()


if __name__ == "__main__":
    main()

