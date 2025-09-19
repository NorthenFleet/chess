#!/usr/bin/env python3
"""
神经网络训练启动脚本

启动中国象棋AI的神经网络训练过程
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ai.trainer import ChessTrainer
from train_config import (
    DEVICE, NETWORK_CONFIG, TRAINING_CONFIG, TRAINING_PHASES,
    LOGGING_CONFIG, PATH_CONFIG, get_current_phase_config,
    update_training_config_for_phase
)

def setup_logging():
    """设置日志配置"""
    os.makedirs(PATH_CONFIG['log_dir'], exist_ok=True)
    
    log_file = os.path.join(PATH_CONFIG['log_dir'], LOGGING_CONFIG['log_file'])
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def create_directories():
    """创建必要的目录"""
    for dir_path in PATH_CONFIG.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

def print_training_info(trainer, total_iterations):
    """打印训练信息"""
    print("\n" + "="*60)
    print("🚀 中国象棋AI神经网络训练")
    print("="*60)
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  训练设备: {DEVICE}")
    print(f"🧠 网络参数: {trainer.network.get_model_info()['total_parameters']:,}")
    print(f"🔄 总迭代数: {total_iterations}")
    print(f"🎮 每轮自对弈: {TRAINING_CONFIG['self_play_games']} 局")
    print(f"🌳 MCTS模拟: {TRAINING_CONFIG['mcts_simulations']} 次")
    print(f"📊 批次大小: {TRAINING_CONFIG['batch_size']}")
    print(f"📈 学习率: {TRAINING_CONFIG['learning_rate']}")
    print("="*60)
    
    # 打印训练阶段信息
    print("\n📋 训练阶段规划:")
    for phase_name, phase_config in TRAINING_PHASES.items():
        print(f"  {phase_name}: {phase_config['name']} ({phase_config['iterations']}轮)")
        print(f"    - {phase_config['description']}")
    print()

def train_with_phases(trainer, total_iterations, logger):
    """分阶段训练"""
    current_iteration = 0
    
    for phase_name, phase_config in TRAINING_PHASES.items():
        if current_iteration >= total_iterations:
            break
            
        phase_iterations = min(phase_config['iterations'], 
                             total_iterations - current_iteration)
        
        logger.info(f"开始 {phase_config['name']} - {phase_iterations} 轮迭代")
        print(f"\n🎯 {phase_config['name']}")
        print(f"📝 {phase_config['description']}")
        print(f"🔄 迭代次数: {phase_iterations}")
        
        # 更新训练配置
        phase_training_config = update_training_config_for_phase(
            TRAINING_CONFIG, phase_config
        )
        trainer.training_config.update(phase_training_config)
        
        # 执行该阶段的训练
        phase_start_time = time.time()
        
        for i in range(phase_iterations):
            iteration_start_time = time.time()
            
            # 执行训练迭代
            stats = trainer.train_iteration()
            
            iteration_time = time.time() - iteration_start_time
            current_iteration += 1
            
            # 打印进度
            if (current_iteration % LOGGING_CONFIG['print_interval']) == 0:
                print(f"迭代 {current_iteration}/{total_iterations} | "
                      f"损失: {stats['total_loss']:.4f} | "
                      f"策略损失: {stats['policy_loss']:.4f} | "
                      f"价值损失: {stats['value_loss']:.4f} | "
                      f"胜率: {stats.get('win_rate', 0):.2%} | "
                      f"用时: {iteration_time:.1f}s")
            
            # 记录日志
            logger.info(f"迭代 {current_iteration} - 损失: {stats['total_loss']:.4f}, "
                       f"胜率: {stats.get('win_rate', 0):.2%}")
        
        phase_time = time.time() - phase_start_time
        logger.info(f"完成 {phase_config['name']} - 用时: {phase_time:.1f}秒")
        print(f"✅ {phase_config['name']} 完成 - 用时: {phase_time:.1f}秒")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动中国象棋AI训练')
    parser.add_argument('--iterations', type=int, default=200, 
                       help='训练迭代次数 (默认: 200)')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--test', action='store_true',
                       help='运行测试模式（少量迭代）')
    
    args = parser.parse_args()
    
    # 测试模式配置
    if args.test:
        print("🧪 测试模式启动")
        args.iterations = 5
        TRAINING_CONFIG.update({
            'self_play_games': 2,
            'mcts_simulations': 100,
            'epochs_per_iteration': 2,
            'batch_size': 8
        })
        NETWORK_CONFIG.update({
            'hidden_channels': 32,
            'num_residual_blocks': 2,
            'use_attention': False
        })
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始训练准备")
    
    # 创建目录
    create_directories()
    
    try:
        # 创建训练器
        print("🔧 初始化训练器...")
        trainer = ChessTrainer(
            network_config=NETWORK_CONFIG,
            training_config=TRAINING_CONFIG,
            device=DEVICE
        )
        
        # 从检查点恢复
        if args.resume:
            print(f"📂 从检查点恢复: {args.resume}")
            trainer.load_checkpoint(args.resume)
            logger.info(f"从检查点恢复: {args.resume}")
        
        # 打印训练信息
        print_training_info(trainer, args.iterations)
        
        # 开始训练
        logger.info(f"开始训练 - 总迭代数: {args.iterations}")
        start_time = time.time()
        
        if args.test:
            # 测试模式：简单训练
            trainer.train(args.iterations, PATH_CONFIG['checkpoint_dir'])
        else:
            # 正常模式：分阶段训练
            train_with_phases(trainer, args.iterations, logger)
        
        # 训练完成
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"\n🎉 训练完成！")
        print(f"⏱️  总用时: {hours}小时 {minutes}分钟 {seconds}秒")
        print(f"💾 模型已保存到: {PATH_CONFIG['checkpoint_dir']}")
        
        logger.info(f"训练完成 - 总用时: {total_time:.1f}秒")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        logger.info("训练被用户中断")
        
        # 保存当前状态
        emergency_checkpoint = os.path.join(
            PATH_CONFIG['checkpoint_dir'], 
            f"emergency_checkpoint_{int(time.time())}.pth"
        )
        if 'trainer' in locals():
            trainer.save_checkpoint(emergency_checkpoint)
            print(f"💾 紧急检查点已保存: {emergency_checkpoint}")
            
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        logger.error(f"训练错误: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()