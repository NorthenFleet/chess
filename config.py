#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋AI训练系统 - 统一配置文件

本文件包含了训练系统的所有配置参数，方便训练人员查询和修改。
配置分为以下几个部分：
1. 系统配置 - 设备、路径等基础设置
2. 神经网络配置 - 网络架构参数
3. MCTS配置 - 蒙特卡洛树搜索参数
4. 训练配置 - 训练超参数和策略
5. 评估配置 - 模型评估相关参数
6. 数据管理配置 - 训练数据管理
7. 日志配置 - 日志记录设置
8. 训练阶段配置 - 分阶段训练策略

修改建议：
- 初学者建议只修改基础参数（如训练轮数、批次大小等）
- 高级用户可以调整网络架构和MCTS参数
- 生产环境建议启用GPU加速和完整的日志记录
"""

import torch
import os
from typing import Dict, Any, Optional

# ============================================================================
# 1. 系统配置
# ============================================================================

class SystemConfig:
    """系统基础配置"""
    
    # 计算设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 文件路径配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'training_data')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    
    # 确保目录存在
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        for dir_path in [cls.CHECKPOINT_DIR, cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# 2. 神经网络配置
# ============================================================================

class NetworkConfig:
    """神经网络架构配置"""
    
    # 网络结构参数
    INPUT_CHANNELS = 14          # 输入通道数（14种棋子类型）
    HIDDEN_CHANNELS = 256        # 隐藏层通道数
    NUM_RESIDUAL_BLOCKS = 8      # 残差块数量
    NUM_ATTENTION_HEADS = 8      # 注意力头数
    ACTION_SPACE_SIZE = 8100     # 动作空间大小（90*90）
    USE_ATTENTION = True         # 是否使用注意力机制
    
    # 训练相关参数
    DROPOUT_RATE = 0.1          # Dropout率
    BATCH_NORM_MOMENTUM = 0.1   # BatchNorm动量
    
    @classmethod
    def to_dict(cls):
        """转换为字典格式，只包含ChessNet构造函数需要的参数"""
        return {
            'input_channels': cls.INPUT_CHANNELS,
            'hidden_channels': cls.HIDDEN_CHANNELS,
            'num_residual_blocks': cls.NUM_RESIDUAL_BLOCKS,
            'num_attention_heads': cls.NUM_ATTENTION_HEADS,
            'action_space_size': cls.ACTION_SPACE_SIZE,
            'use_attention': cls.USE_ATTENTION
        }


# ============================================================================
# 3. MCTS配置
# ============================================================================

class MCTSConfig:
    """蒙特卡洛树搜索配置"""
    
    # 基础MCTS参数
    NUM_SIMULATIONS = 800            # MCTS模拟次数（影响搜索深度和质量）
    C_PUCT = 1.4                     # UCB常数（平衡探索与利用）
    TEMPERATURE = 1.0                # 温度参数（控制动作选择随机性）
    
    # 噪声配置（增加探索性）
    ADD_NOISE = True                 # 是否添加狄利克雷噪声
    NOISE_ALPHA = 0.3                # 噪声强度参数
    NOISE_EPSILON = 0.25             # 噪声混合比例
    
    # 搜索优化
    VIRTUAL_LOSS = 3                 # 虚拟损失（并行搜索优化）
    MAX_SEARCH_DEPTH = 100           # 最大搜索深度
    
    # 对战时的MCTS配置（相对保守）
    GAME_NUM_SIMULATIONS = 400       # 人机对战时的模拟次数
    GAME_TEMPERATURE = 0.1           # 人机对战时的温度（更确定性）
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'num_simulations': cls.NUM_SIMULATIONS,
            'c_puct': cls.C_PUCT,
            'temperature': cls.TEMPERATURE,
            'add_noise': cls.ADD_NOISE,
            'noise_alpha': cls.NOISE_ALPHA,
            'noise_epsilon': cls.NOISE_EPSILON,
            'virtual_loss': cls.VIRTUAL_LOSS,
            'max_search_depth': cls.MAX_SEARCH_DEPTH,
        }


# ============================================================================
# 4. 训练配置
# ============================================================================

class TrainingConfig:
    """训练超参数配置"""
    
    # 基础训练参数
    LEARNING_RATE = 0.001            # 学习率（影响收敛速度）
    BATCH_SIZE = 32                  # 批次大小（影响内存使用和收敛稳定性）
    EPOCHS_PER_ITERATION = 10        # 每次迭代的训练轮数
    WEIGHT_DECAY = 1e-4              # 权重衰减（L2正则化）
    
    # 学习率调度
    USE_LR_SCHEDULER = True          # 是否使用学习率调度
    LR_SCHEDULER_TYPE = 'step'       # 调度器类型 ('step', 'cosine', 'exponential')
    LR_SCHEDULER_STEP = 50           # 学习率调度步长
    LR_SCHEDULER_GAMMA = 0.9         # 学习率衰减因子
    
    # 优化器配置
    OPTIMIZER = 'adam'               # 优化器类型 ('adam', 'sgd', 'adamw')
    MOMENTUM = 0.9                   # SGD动量（仅SGD使用）
    BETA1 = 0.9                      # Adam beta1参数
    BETA2 = 0.999                    # Adam beta2参数
    
    # 损失函数权重
    POLICY_LOSS_WEIGHT = 1.0         # 策略损失权重
    VALUE_LOSS_WEIGHT = 1.0          # 价值损失权重
    
    # 梯度裁剪
    GRADIENT_CLIPPING = True         # 是否使用梯度裁剪
    MAX_GRAD_NORM = 1.0              # 最大梯度范数
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'learning_rate': cls.LEARNING_RATE,
            'batch_size': cls.BATCH_SIZE,
            'epochs_per_iteration': cls.EPOCHS_PER_ITERATION,
            'weight_decay': cls.WEIGHT_DECAY,
            'use_lr_scheduler': cls.USE_LR_SCHEDULER,
            'lr_scheduler_type': cls.LR_SCHEDULER_TYPE,
            'lr_scheduler_step': cls.LR_SCHEDULER_STEP,
            'lr_scheduler_gamma': cls.LR_SCHEDULER_GAMMA,
            'optimizer': cls.OPTIMIZER,
            'momentum': cls.MOMENTUM,
            'beta1': cls.BETA1,
            'beta2': cls.BETA2,
            'policy_loss_weight': cls.POLICY_LOSS_WEIGHT,
            'value_loss_weight': cls.VALUE_LOSS_WEIGHT,
            'gradient_clipping': cls.GRADIENT_CLIPPING,
            'max_grad_norm': cls.MAX_GRAD_NORM,
        }


# ============================================================================
# 5. 自对弈配置
# ============================================================================

class SelfPlayConfig:
    """自对弈配置"""
    
    # 自对弈基础参数
    GAMES_PER_ITERATION = 10         # 每次迭代的自对弈游戏数
    MAX_GAME_LENGTH = 200            # 最大游戏长度（防止无限循环）
    
    # 游戏结束条件
    DRAW_THRESHOLD = 100             # 平局判定阈值（回合数）
    REPETITION_THRESHOLD = 3         # 重复局面判定阈值
    
    # 并行配置
    NUM_WORKERS = 1                  # 并行工作进程数（建议不超过CPU核心数）
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'games_per_iteration': cls.GAMES_PER_ITERATION,
            'max_game_length': cls.MAX_GAME_LENGTH,
            'draw_threshold': cls.DRAW_THRESHOLD,
            'repetition_threshold': cls.REPETITION_THRESHOLD,
            'num_workers': cls.NUM_WORKERS,
        }


# ============================================================================
# 6. 评估配置
# ============================================================================

class EvaluationConfig:
    """模型评估配置"""
    
    # 评估基础参数
    EVAL_GAMES = 20                  # 评估游戏数量
    EVAL_INTERVAL = 5                # 评估间隔（每N次迭代评估一次）
    
    # 评估对手配置
    EVAL_MCTS_SIMULATIONS = 400      # 评估时的MCTS模拟次数
    EVAL_TEMPERATURE = 0.1           # 评估时的温度参数
    
    # 性能指标
    WIN_RATE_THRESHOLD = 0.55        # 模型更新的胜率阈值
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'eval_games': cls.EVAL_GAMES,
            'eval_interval': cls.EVAL_INTERVAL,
            'eval_mcts_simulations': cls.EVAL_MCTS_SIMULATIONS,
            'eval_temperature': cls.EVAL_TEMPERATURE,
            'win_rate_threshold': cls.WIN_RATE_THRESHOLD,
        }


# ============================================================================
# 7. 数据管理配置
# ============================================================================

class DataConfig:
    """训练数据管理配置"""
    
    # 数据集大小管理
    MAX_EXAMPLES = 50000             # 最大训练样本数
    MIN_EXAMPLES = 1000              # 最小训练样本数
    
    # 数据清理策略
    DATA_CLEANUP_INTERVAL = 100      # 数据清理间隔
    KEEP_RECENT_EXAMPLES = 30000     # 保留最近的样本数
    
    # 数据增强
    USE_DATA_AUGMENTATION = False    # 是否使用数据增强
    AUGMENTATION_RATIO = 0.2         # 数据增强比例
    
    # 数据保存
    SAVE_TRAINING_DATA = True        # 是否保存训练数据
    DATA_SAVE_INTERVAL = 50          # 数据保存间隔
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'max_examples': cls.MAX_EXAMPLES,
            'min_examples': cls.MIN_EXAMPLES,
            'data_cleanup_interval': cls.DATA_CLEANUP_INTERVAL,
            'keep_recent_examples': cls.KEEP_RECENT_EXAMPLES,
            'use_data_augmentation': cls.USE_DATA_AUGMENTATION,
            'augmentation_ratio': cls.AUGMENTATION_RATIO,
            'save_training_data': cls.SAVE_TRAINING_DATA,
            'data_save_interval': cls.DATA_SAVE_INTERVAL,
        }


# ============================================================================
# 8. 日志配置
# ============================================================================

class LoggingConfig:
    """日志记录配置"""
    
    # 基础日志配置
    LOG_LEVEL = 'INFO'               # 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    LOG_FILE = 'training.log'        # 日志文件名
    
    # 控制台输出
    PRINT_INTERVAL = 10              # 控制台打印间隔
    VERBOSE = True                   # 是否详细输出
    
    # TensorBoard配置
    USE_TENSORBOARD = False          # 是否使用TensorBoard
    TENSORBOARD_DIR = 'runs'         # TensorBoard日志目录
    
    # 性能监控
    MONITOR_GPU = True               # 是否监控GPU使用情况
    MONITOR_MEMORY = True            # 是否监控内存使用情况
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'log_level': cls.LOG_LEVEL,
            'log_file': cls.LOG_FILE,
            'print_interval': cls.PRINT_INTERVAL,
            'verbose': cls.VERBOSE,
            'use_tensorboard': cls.USE_TENSORBOARD,
            'tensorboard_dir': cls.TENSORBOARD_DIR,
            'monitor_gpu': cls.MONITOR_GPU,
            'monitor_memory': cls.MONITOR_MEMORY,
        }


# ============================================================================
# 9. 检查点配置
# ============================================================================

class CheckpointConfig:
    """模型检查点配置"""
    
    # 检查点保存策略
    SAVE_INTERVAL = 10               # 检查点保存间隔
    MAX_CHECKPOINTS = 5              # 最大检查点数量
    
    # 最佳模型保存
    SAVE_BEST_MODEL = True           # 是否保存最佳模型
    BEST_MODEL_METRIC = 'win_rate'   # 最佳模型评判指标
    
    # 检查点文件命名
    CHECKPOINT_PREFIX = 'checkpoint' # 检查点文件前缀
    BEST_MODEL_NAME = 'best_model.pth'  # 最佳模型文件名
    FINAL_MODEL_NAME = 'final_model.pth'  # 最终模型文件名
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'save_interval': cls.SAVE_INTERVAL,
            'max_checkpoints': cls.MAX_CHECKPOINTS,
            'save_best_model': cls.SAVE_BEST_MODEL,
            'best_model_metric': cls.BEST_MODEL_METRIC,
            'checkpoint_prefix': cls.CHECKPOINT_PREFIX,
            'best_model_name': cls.BEST_MODEL_NAME,
            'final_model_name': cls.FINAL_MODEL_NAME,
        }


# ============================================================================
# 10. 训练阶段配置
# ============================================================================

class TrainingPhases:
    """分阶段训练配置
    
    训练分为三个阶段：
    1. 基础训练阶段：建立基础棋局理解能力
    2. 强化训练阶段：提升策略质量和搜索深度  
    3. 精英训练阶段：达到专业水平
    """
    
    PHASES = {
        'phase_1': {
            'name': '基础训练阶段',
            'iterations': 50,                # 训练迭代次数
            'mcts_simulations': 400,         # MCTS模拟次数
            'self_play_games': 5,            # 自对弈游戏数
            'temperature': 1.2,              # 温度参数（较高，增加探索）
            'learning_rate': 0.001,          # 学习率
            'batch_size': 16,                # 批次大小
            'description': '建立基础棋局理解能力，学习基本走法和规则'
        },
        'phase_2': {
            'name': '强化训练阶段',
            'iterations': 100,               # 训练迭代次数
            'mcts_simulations': 800,         # MCTS模拟次数
            'self_play_games': 10,           # 自对弈游戏数
            'temperature': 1.0,              # 温度参数（中等）
            'learning_rate': 0.0005,         # 学习率（降低）
            'batch_size': 32,                # 批次大小
            'description': '提升策略质量和搜索深度，学习复杂战术'
        },
        'phase_3': {
            'name': '精英训练阶段',
            'iterations': 50,                # 训练迭代次数
            'mcts_simulations': 1200,        # MCTS模拟次数（最高）
            'self_play_games': 15,           # 自对弈游戏数
            'temperature': 0.8,              # 温度参数（较低，更确定性）
            'learning_rate': 0.0001,         # 学习率（最低）
            'batch_size': 64,                # 批次大小（最大）
            'description': '达到专业水平，精细调优策略和战术理解'
        }
    }
    
    @classmethod
    def get_phase_config(cls, iteration: int) -> Dict[str, Any]:
        """根据迭代次数获取当前阶段配置"""
        total_iterations = 0
        for phase_name, phase_config in cls.PHASES.items():
            total_iterations += phase_config['iterations']
            if iteration <= total_iterations:
                return phase_config
        
        # 如果超出所有阶段，返回最后一个阶段的配置
        return cls.PHASES['phase_3']
    
    @classmethod
    def get_current_phase_name(cls, iteration: int) -> str:
        """获取当前阶段名称"""
        return cls.get_phase_config(iteration)['name']


# ============================================================================
# 11. 配置管理器
# ============================================================================

class ConfigManager:
    """配置管理器
    
    提供统一的配置访问接口和配置验证功能
    """
    
    def __init__(self):
        """初始化配置管理器"""
        # 确保必要的目录存在
        SystemConfig.ensure_directories()
        
        # 打印系统信息
        self._print_system_info()
    
    def _print_system_info(self):
        """打印系统信息"""
        print("="*60)
        print("🚀 中国象棋AI训练系统配置")
        print("="*60)
        print(f"📱 计算设备: {SystemConfig.DEVICE}")
        print(f"🧠 网络架构: {NetworkConfig.NUM_RESIDUAL_BLOCKS}层残差网络")
        print(f"🎯 动作空间: {NetworkConfig.ACTION_SPACE_SIZE}")
        print(f"🔍 MCTS模拟: {MCTSConfig.NUM_SIMULATIONS}次")
        print(f"📚 批次大小: {TrainingConfig.BATCH_SIZE}")
        print(f"📈 学习率: {TrainingConfig.LEARNING_RATE}")
        print("="*60)
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            'system': {
                'device': SystemConfig.DEVICE,
                'checkpoint_dir': SystemConfig.CHECKPOINT_DIR,
                'data_dir': SystemConfig.DATA_DIR,
                'model_dir': SystemConfig.MODEL_DIR,
                'log_dir': SystemConfig.LOG_DIR,
            },
            'network': NetworkConfig.to_dict(),
            'mcts': MCTSConfig.to_dict(),
            'training': TrainingConfig.to_dict(),
            'self_play': SelfPlayConfig.to_dict(),
            'evaluation': EvaluationConfig.to_dict(),
            'data': DataConfig.to_dict(),
            'logging': LoggingConfig.to_dict(),
            'checkpoint': CheckpointConfig.to_dict(),
            'phases': TrainingPhases.PHASES,
        }
    
    def validate_config(self) -> bool:
        """验证配置的合理性"""
        try:
            # 验证网络配置
            assert NetworkConfig.INPUT_CHANNELS > 0, "输入通道数必须大于0"
            assert NetworkConfig.HIDDEN_CHANNELS > 0, "隐藏层通道数必须大于0"
            assert NetworkConfig.NUM_RESIDUAL_BLOCKS > 0, "残差块数量必须大于0"
            assert NetworkConfig.ACTION_SPACE_SIZE > 0, "动作空间大小必须大于0"
            
            # 验证训练配置
            assert TrainingConfig.LEARNING_RATE > 0, "学习率必须大于0"
            assert TrainingConfig.BATCH_SIZE > 0, "批次大小必须大于0"
            assert TrainingConfig.EPOCHS_PER_ITERATION > 0, "每次迭代的训练轮数必须大于0"
            
            # 验证MCTS配置
            assert MCTSConfig.NUM_SIMULATIONS > 0, "MCTS模拟次数必须大于0"
            assert MCTSConfig.C_PUCT > 0, "UCB常数必须大于0"
            assert MCTSConfig.TEMPERATURE >= 0, "温度参数必须非负"
            
            print("✅ 配置验证通过")
            return True
            
        except AssertionError as e:
            print(f"❌ 配置验证失败: {e}")
            return False
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        import json
        config = self.get_all_configs()
        
        # 处理不能JSON序列化的对象
        def convert_to_serializable(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, type):
                return str(obj)
            else:
                return obj
        
        # 递归转换配置
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_to_serializable(data)
        
        serializable_config = recursive_convert(config)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置已保存到: {filepath}")


# ============================================================================
# 12. 兼容性接口（保持向后兼容）
# ============================================================================

# 为了保持与现有代码的兼容性，提供旧的配置变量
DEVICE = SystemConfig.DEVICE
NETWORK_CONFIG = NetworkConfig.to_dict()
MCTS_CONFIG = MCTSConfig.to_dict()
TRAINING_CONFIG = TrainingConfig.to_dict()
TRAINING_PHASES = TrainingPhases.PHASES
LOGGING_CONFIG = LoggingConfig.to_dict()
PATH_CONFIG = {
    'checkpoint_dir': SystemConfig.CHECKPOINT_DIR,
    'data_dir': SystemConfig.DATA_DIR,
    'model_dir': SystemConfig.MODEL_DIR,
    'log_dir': SystemConfig.LOG_DIR,
}


# ============================================================================
# 13. 主程序入口
# ============================================================================

def main():
    """主程序入口 - 用于测试和展示配置"""
    config_manager = ConfigManager()
    
    # 验证配置
    if config_manager.validate_config():
        print("\n📋 当前配置概览:")
        print("-" * 40)
        
        # 显示关键配置信息
        print(f"🎯 训练阶段数: {len(TrainingPhases.PHASES)}")
        print(f"🔄 总训练迭代: {sum(phase['iterations'] for phase in TrainingPhases.PHASES.values())}")
        print(f"💾 最大样本数: {DataConfig.MAX_EXAMPLES}")
        print(f"📊 评估间隔: {EvaluationConfig.EVAL_INTERVAL}")
        print(f"💾 检查点间隔: {CheckpointConfig.SAVE_INTERVAL}")
        
        # 保存配置到文件
        config_file = os.path.join(SystemConfig.LOG_DIR, 'current_config.json')
        config_manager.save_config(config_file)
        
        print(f"\n📝 完整配置已保存到: {config_file}")
        print("🎉 配置系统初始化完成！")
    else:
        print("❌ 配置验证失败，请检查配置参数")


if __name__ == "__main__":
    main()