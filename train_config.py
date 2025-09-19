"""
神经网络训练配置文件

包含网络架构配置、训练超参数和设备设置
"""

import torch

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}")

# 网络架构配置
NETWORK_CONFIG = {
    'input_channels': 14,           # 输入通道数（棋盘编码）
    'hidden_channels': 128,         # 隐藏层通道数
    'num_residual_blocks': 8,       # 残差块数量
    'num_attention_heads': 8,       # 注意力头数量
    'action_space_size': 8100,      # 动作空间大小
    'use_attention': True,          # 是否使用自注意力机制
}

# 训练配置
TRAINING_CONFIG = {
    # MCTS配置
    'mcts_simulations': 800,        # MCTS模拟次数
    'c_puct': 1.4,                  # UCB常数
    'temperature': 1.0,             # 温度参数
    'add_noise': True,              # 是否添加狄利克雷噪声
    'noise_alpha': 0.3,             # 噪声参数
    
    # 自对弈配置
    'self_play_games': 10,          # 每次迭代的自对弈游戏数
    'max_game_length': 200,         # 最大游戏长度
    
    # 训练配置
    'epochs_per_iteration': 10,     # 每次迭代的训练轮数
    'batch_size': 32,               # 批次大小
    'learning_rate': 0.001,         # 学习率
    'weight_decay': 1e-4,           # 权重衰减
    'lr_scheduler_step': 50,        # 学习率调度步长
    'lr_scheduler_gamma': 0.9,      # 学习率衰减因子
    
    # 评估配置
    'eval_games': 20,               # 评估游戏数
    'eval_interval': 5,             # 评估间隔
    
    # 检查点配置
    'checkpoint_interval': 10,      # 检查点保存间隔
    'max_checkpoints': 5,           # 最大检查点数量
    
    # 数据管理
    'max_examples': 50000,          # 最大训练样本数
    'min_examples': 1000,           # 最小训练样本数
}

# 训练阶段配置
TRAINING_PHASES = {
    'phase_1': {
        'name': '基础训练阶段',
        'iterations': 50,
        'mcts_simulations': 400,
        'self_play_games': 5,
        'temperature': 1.2,
        'description': '建立基础棋局理解能力'
    },
    'phase_2': {
        'name': '强化训练阶段', 
        'iterations': 100,
        'mcts_simulations': 800,
        'self_play_games': 10,
        'temperature': 1.0,
        'description': '提升策略质量和搜索深度'
    },
    'phase_3': {
        'name': '精英训练阶段',
        'iterations': 50,
        'mcts_simulations': 1200,
        'self_play_games': 15,
        'temperature': 0.8,
        'description': '达到专业水平'
    }
}

# 日志配置
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'training.log',
    'tensorboard_dir': 'runs',
    'print_interval': 10,           # 打印间隔
}

# 文件路径配置
PATH_CONFIG = {
    'checkpoint_dir': 'checkpoints',
    'data_dir': 'training_data',
    'model_dir': 'models',
    'log_dir': 'logs',
}

def get_current_phase_config(iteration: int) -> dict:
    """
    根据当前迭代次数获取训练阶段配置
    
    Args:
        iteration: 当前迭代次数
        
    Returns:
        当前阶段的配置字典
    """
    if iteration <= TRAINING_PHASES['phase_1']['iterations']:
        return TRAINING_PHASES['phase_1']
    elif iteration <= TRAINING_PHASES['phase_1']['iterations'] + TRAINING_PHASES['phase_2']['iterations']:
        return TRAINING_PHASES['phase_2']
    else:
        return TRAINING_PHASES['phase_3']

def update_training_config_for_phase(config: dict, phase_config: dict) -> dict:
    """
    根据训练阶段更新训练配置
    
    Args:
        config: 基础训练配置
        phase_config: 阶段配置
        
    Returns:
        更新后的训练配置
    """
    updated_config = config.copy()
    updated_config.update({
        'mcts_simulations': phase_config['mcts_simulations'],
        'self_play_games': phase_config['self_play_games'],
        'temperature': phase_config['temperature'],
    })
    return updated_config

if __name__ == "__main__":
    print("训练配置信息:")
    print(f"设备: {DEVICE}")
    print(f"网络配置: {NETWORK_CONFIG}")
    print(f"训练配置: {TRAINING_CONFIG}")
    print(f"训练阶段: {list(TRAINING_PHASES.keys())}")