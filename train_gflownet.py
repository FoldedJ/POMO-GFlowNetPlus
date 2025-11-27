import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging

from ImpTSPEnv import TSPEnv_Improve
from GFlowNetTSPModel import GFlowNetTSPModel
from GFlowNetAgent import GFlowNetAgent
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold

def train(
    batch_size=16,
    problem_size=20,
    max_steps=None, # 默认设置为 problem_size * 2
    embedding_dim=128,
    head_num=8,
    lr=1e-4,
    epochs=100,
    eval_interval=10,
    save_interval=10, # 每10个epoch保存一次模型
    save_dir="./checkpoints", # 模型保存路径
    encoder_layer_num=1, #6
    ff_hidden_dim=512, # embedding_dim * 4
    log_dir="./logs", # 日志保存路径
    lambda_v=1.0, # 价值损失权重
    lambda_pc=0.1 # 策略一致性损失权重
):
    if max_steps is None:
        max_steps = problem_size * 2

    # 配置日志
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

    # 模型参数
    model_params = {
        'embedding_dim': embedding_dim,
        'head_num': head_num,
        'qkv_dim': embedding_dim // head_num,
        'encoder_layer_num': encoder_layer_num,
        'ff_hidden_dim': ff_hidden_dim,
        'sqrt_embedding_dim': embedding_dim**0.5
    }

    # 1. 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. 初始化环境、模型和智能体
    env = TSPEnv_Improve(device=device, problem_size=problem_size, max_steps=max_steps)
    model = GFlowNetTSPModel(**model_params).to(device)
    agent = GFlowNetAgent(env, model).to(device)

    # 3. 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. 训练循环
    model.train() # 设置模型为训练模式
    for epoch in range(epochs):
        # 生成新的训练数据
        problems = get_random_problems(batch_size, problem_size).to(device)
        problems = augment_xy_data_by_8_fold(problems)
        
        # 运行一个episode来收集轨迹数据
        episode_states, episode_rewards_per_step, episode_action_log_probs, episode_value_preds = agent.run_episode(problems)

        # 计算GFlowNet损失
        gflownet_loss = 0.0
        value_loss = 0.0
        policy_consistency_loss = 0.0
        
        # 最终状态是episode_states中的最后一个
        final_state = episode_states[-1]
        final_path_length = final_state.path_length # (batch,)

        # 确保final_path_length是正数，避免log(0)或log(负数)
        final_path_length = torch.clamp(final_path_length, min=1e-6)
        log_final_path_length = torch.log(final_path_length)

        # 计算轨迹中每一步的损失
        for t in range(len(episode_action_log_probs)):
            # GFlowNet Detailed Balance Loss
            log_F_s_t = -torch.log(torch.clamp(episode_value_preds[t], min=1e-6)) 
            action_log_prob_t = episode_action_log_probs[t]

            if t < len(episode_action_log_probs) - 1: # 非终止步
                log_F_s_t_plus_1 = -torch.log(torch.clamp(episode_value_preds[t+1], min=1e-6))
                gflownet_loss_t = (log_F_s_t + action_log_prob_t - log_F_s_t_plus_1).pow(2).mean()
            else: # 终止步
                gflownet_loss_t = (log_F_s_t + action_log_prob_t - (-log_final_path_length)).pow(2).mean()
            gflownet_loss += gflownet_loss_t

            # Value Loss (预测的路径长度与实际最终路径长度的L2损失)
            value_loss_t = (episode_value_preds[t] - final_path_length).pow(2).mean()
            value_loss += value_loss_t

            # Policy Consistency Loss (可选，这里简化为动作概率与价值预测的一致性)
            # 假设我们希望动作概率与价值预测呈正相关
            # 这是一个简化的示例，实际可能需要更复杂的定义
            policy_consistency_loss_t = (action_log_prob_t + log_F_s_t).pow(2).mean() # 鼓励动作概率与价值预测一致
            policy_consistency_loss += policy_consistency_loss_t

        # 平均损失
        gflownet_loss /= len(episode_action_log_probs)
        value_loss /= len(episode_action_log_probs)
        policy_consistency_loss /= len(episode_action_log_probs)

        # 总损失
        total_loss = gflownet_loss + lambda_v * value_loss + lambda_pc * policy_consistency_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 记录和打印训练信息
        avg_episode_reward = torch.sum(torch.stack(episode_rewards_per_step), dim=0).mean()
        avg_final_path_length = final_path_length.mean()

        logging.info(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}, GFlowNet Loss: {gflownet_loss:.4f}, Value Loss: {value_loss:.4f}, Policy Consistency Loss: {policy_consistency_loss:.4f}, Avg Episode Reward: {avg_episode_reward:.4f}, Avg Final Path Length: {avg_final_path_length:.4f}")

        # 每隔eval_interval个epoch进行一次评估
        if (epoch + 1) % eval_interval == 0:
            evaluate(env, agent, device)

        # 每隔save_interval个epoch保存一次模型
        if (epoch + 1) % save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"gflownet_tsp_epoch_{epoch+1}.pt")
            save_model(model, optimizer, epoch, model_path)

    logging.info("训练完成！")

def evaluate(env, agent, device, num_problems=100):
    logging.info(f"开始评估模型，问题数量: {num_problems}")
    agent.model.eval() # 设置模型为评估模式
    
    with torch.no_grad(): # 评估时不需要计算梯度
        test_problems = get_random_problems(num_problems, env.problem_size).to(device)
        
        # 运行episode收集轨迹数据
        episode_states, _, _, _ = agent.run_episode(test_problems)
        
        final_state = episode_states[-1]
        final_path_length = final_state.path_length
        
        avg_final_path_length = final_path_length.mean().item()
        logging.info(f"评估完成，平均最终路径长度: {avg_final_path_length:.4f}")
        
    agent.model.train() # 评估结束后，将模型设置回训练模式
    return avg_final_path_length

def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    logging.info(f"模型已保存到 {path} (Epoch {epoch})")

def load_model(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logging.info(f"模型已从 {path} 加载 (Epoch {epoch})")
    return epoch

if __name__ == '__main__':
    train()
