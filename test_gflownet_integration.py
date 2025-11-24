import torch
from ImpTSPEnv import TSPEnv_Improve
from GFlowNetTSPModel import GFlowNetTSPModel
from GFlowNetAgent import GFlowNetAgent

# 1. 定义环境和模型参数
problem_size = 20
batch_size = 2
embedding_dim = 128
head_num = 8
env_params = {
    'problem_size': problem_size,
    'max_steps': problem_size * 2
}
model_params = {
    'embedding_dim': embedding_dim,
    'head_num': head_num,
    'qkv_dim': embedding_dim // head_num,
    'encoder_layer_num': 3,
    'ff_hidden_dim': embedding_dim * 4,
    'sqrt_embedding_dim': embedding_dim**0.5
}

# 2. 实例化环境
env = TSPEnv_Improve(**env_params)
env.load_problems(batch_size)

# 3. 实例化模型
model = GFlowNetTSPModel(**model_params)

# 4. 实例化代理
agent = GFlowNetAgent(env, model)

# 5. 运行一个episode
print("开始运行一个episode...")
episode_states, episode_rewards = agent.run_episode()

# 6. 打印结果
print("Episode运行结束。")

initial_state = episode_states[0]
final_state = episode_states[-1]

print(f"初始路径长度: {initial_state.path_length.mean().item():.4f}")
print(f"最终路径长度: {final_state.path_length.mean().item():.4f}")
print(f"总奖励: {torch.stack(episode_rewards).sum(dim=0).mean().item():.4f}")

print("集成测试完成。")