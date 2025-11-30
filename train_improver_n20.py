##########################################################################################
# Machine Environment Config

DEBUG_MODE = False  # 调试模式（减少训练规模与频率）
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
from utils.utils import create_logger

from improver.GFlowTrainerCore import GFlowTBTrainer, TBConfig  # 导入TB训练器与配置

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

trainer_params = {  # 训练流程参数
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 1000,
    'logging': {
        'model_save_interval': 100,
        'img_save_file': 'style_loss_1.json'
    },
}

tb_cfg = TBConfig(temperature=1.0, lr=1e-4, weight_decay=1e-6,
                  train_batch_size=64, train_episodes=10000, save_interval=trainer_params['logging']['model_save_interval'])  # TB超参数

logger_params = {  # 日志配置
    'log_file': {
        'desc': 'train__improver_tsp_n20_tb',
        'filename': 'run_log'
    }
}

##########################################################################################
# main

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

    trainer = GFlowTBTrainer(model_params=model_params, tb_cfg=tb_cfg)  # 构建训练器
    save_dir = os.path.join('./result', 'improver_tsp20_tb')  # 模型与曲线保存目录
    trainer.run(save_dir=save_dir, epochs=trainer_params['epochs'], img_style_file=trainer_params['logging']['img_save_file'])  # 开始训练


def _set_debug_mode():
    global trainer_params, tb_cfg
    trainer_params['epochs'] = 10  # 降低epoch
    tb_cfg.train_batch_size = 8  # 降低batch
    tb_cfg.save_interval = 5  # 频繁保存


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]  # 打印各类参数

##########################################################################################

if __name__ == "__main__":
    main()
