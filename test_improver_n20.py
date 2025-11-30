##########################################################################################
# Machine Environment Config

DEBUG_MODE = False  # 调试模式（减少测试规模）
USE_CUDA = not DEBUG_MODE  # 是否使用GPU
CUDA_DEVICE_NUM = 0  # GPU设备编号

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 将工作目录切到脚本所在位置
sys.path.insert(0, "..")  # 方便导入上层包
sys.path.insert(0, "../..")

##########################################################################################
# import

import logging
from utils.utils import create_logger, util_save_log_image_with_label, LogData

import torch
from improver.GFlowImprover import GFlowImprover  # 单次改进器
from improver.ImproverModelParts import compute_tour_length
from TSPModel import TSPModel
from TSPEnv import TSPEnv

##########################################################################################
# parameters

model_params = {  # 模型结构参数（与原POMO一致）
    'problem_size': 20,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {  # 测试流程参数
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': 200,
    'test_batch_size': 64,
    'augmentation_enable': False,
}

logger_params = {
    'log_file': {
        'desc': 'test__improver_tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def build_initial_tour(problems: torch.Tensor) -> torch.Tensor:
    """生成随机初始路径作为提升式算法的基础路径"""
    device = problems.device
    batch = problems.size(0)
    N = problems.size(1)
    tours = torch.stack([torch.randperm(N, device=device) for _ in range(batch)], dim=0)
    return tours


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)  # 初始化日志器
    _print_config()  # 打印配置

    if USE_CUDA:  # 设备设置（新API）
        import torch
        torch.set_default_device(f'cuda:{CUDA_DEVICE_NUM}')
        torch.set_default_dtype(torch.float32)
    else:
        import torch
        torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float32)

    improver = GFlowImprover(model_params, k_backtrack=3, m_reconstruct=3, temperature=1.0)  # 构建改进器
    log = LogData()  # 测试日志

    for ep in range(1, tester_params['test_episodes'] + 1):
        problems = torch.rand(tester_params['test_batch_size'], model_params['problem_size'], 2)  # 随机TSP
        initial = build_initial_tour(problems)  # 基线初始解
        result = improver.improve_once(problems, initial)  # 一次改进
        avg_init = result['initial_lengths'].mean().item()
        avg_best = result['best_lengths'].mean().item()
        log.append('avg_init_len', avg_init)  # 记录初始长度
        log.append('avg_best_len', avg_best)  # 记录改进后长度

    util_save_log_image_with_label(
        result_file_prefix='./result/test_improver',
        img_params={'json_foldername': 'log_image_style', 'filename': 'style_tsp_20.json'},
        result_log=log,
        labels=['avg_init_len', 'avg_best_len']  # 输出对比曲线
    )


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 5  # 减少次数
    tester_params['test_batch_size'] = 8  # 减少批量


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]  # 打印各类参数

##########################################################################################

if __name__ == "__main__":
    main()
