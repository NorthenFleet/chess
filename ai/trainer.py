"""
中国象棋AI训练框架

包含自对弈数据生成、神经网络训练和模型评估
"""

import os
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.game_state import GameState
from ai.network import ChessNet, create_chess_net
from ai.encoder import BoardEncoder, ActionEncoder
from ai.mcts import MCTS


@dataclass
class TrainingExample:
    """训练样本数据结构"""
    board_state: np.ndarray      # 棋盘状态编码
    action_probs: np.ndarray     # MCTS动作概率分布
    value: float                 # 游戏结果价值
    player: str                  # 当前玩家 ('red' 或 'black')


class ChessDataset(Dataset):
    """象棋训练数据集"""
    
    def __init__(self, examples: List[TrainingExample]):
        """
        初始化数据集
        
        Args:
            examples: 训练样本列表
        """
        self.examples = examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[idx]
        
        board_tensor = torch.FloatTensor(example.board_state)
        action_tensor = torch.FloatTensor(example.action_probs)
        value_tensor = torch.FloatTensor([example.value])
        
        return board_tensor, action_tensor, value_tensor


class ChessTrainer:
    """象棋AI训练器"""
    
    def __init__(self,
                 network_config: Dict = None,
                 training_config: Dict = None,
                 device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            network_config: 网络配置
            training_config: 训练配置
            device: 计算设备
        """
        self.device = torch.device(device)
        
        # 默认网络配置
        default_network_config = {
            'input_channels': 14,
            'hidden_channels': 256,
            'num_residual_blocks': 20,
            'num_attention_heads': 8,
            'action_space_size': 8100,
            'use_attention': True
        }
        if network_config:
            default_network_config.update(network_config)
        
        # 默认训练配置
        default_training_config = {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'epochs_per_iteration': 10,
            'mcts_simulations': 800,
            'self_play_games': 100,
            'temperature_threshold': 30,  # 前30步使用温度采样
            'max_game_length': 500,
            'buffer_size': 100000,
            'checkpoint_interval': 10,
            'evaluation_games': 20
        }
        if training_config:
            default_training_config.update(training_config)
        
        self.network_config = default_network_config
        self.training_config = default_training_config
        
        # 创建网络
        self.network = create_chess_net(self.network_config).to(self.device)
        self.target_network = create_chess_net(self.network_config).to(self.device)
        
        # 创建编码器
        self.board_encoder = BoardEncoder()
        self.action_encoder = ActionEncoder()
        
        # 创建MCTS
        self.mcts = MCTS(
            neural_network=self.network,
            board_encoder=self.board_encoder,
            action_encoder=self.action_encoder,
            num_simulations=self.training_config['mcts_simulations']
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # 训练数据缓冲区
        self.training_buffer = deque(maxlen=self.training_config['buffer_size'])
        
        # 训练统计
        self.iteration = 0
        self.total_games = 0
        self.training_history = []
    
    def self_play_game(self) -> List[TrainingExample]:
        """
        执行一局自对弈游戏
        
        Returns:
            训练样本列表
        """
        game_state = GameState()
        game_state.setup_players("AI_Red", "AI_Black")
        game_state.start_game()
        examples = []
        move_count = 0
        
        while not game_state.game_over and move_count < self.training_config['max_game_length']:
            # 设置温度参数
            temperature = 1.0 if move_count < self.training_config['temperature_threshold'] else 0.1
            self.mcts.temperature = temperature
            
            # 获取训练数据
            board_encoded, action_probs, _ = self.mcts.get_training_data(game_state)
            
            # 保存训练样本
            example = TrainingExample(
                board_state=board_encoded,
                action_probs=action_probs,
                value=0.0,  # 稍后更新
                player=game_state.current_player.side
            )
            examples.append(example)
            
            # 根据概率分布选择动作
            if temperature == 0:
                # 贪婪选择
                action_idx = np.argmax(action_probs)
            else:
                # 概率采样
                action_idx = np.random.choice(len(action_probs), p=action_probs)
            
            # 执行动作
            from_pos, to_pos = self.action_encoder.decode_action(action_idx)
            success = game_state.make_move(from_pos, to_pos)
            
            if not success:
                # 非法移动，游戏异常结束
                break
            
            move_count += 1
        
        # 更新训练样本的价值
        winner = game_state.get_winner()
        for example in examples:
            if winner is None:
                example.value = 0.0  # 平局
            elif winner == example.player:
                example.value = 1.0  # 获胜
            else:
                example.value = -1.0  # 失败
        
        return examples
    
    def generate_self_play_data(self, num_games: int) -> List[TrainingExample]:
        """
        生成自对弈训练数据
        
        Args:
            num_games: 游戏局数
            
        Returns:
            训练样本列表
        """
        all_examples = []
        
        print(f"开始生成 {num_games} 局自对弈数据...")
        
        for game_idx in range(num_games):
            if (game_idx + 1) % 10 == 0:
                print(f"已完成 {game_idx + 1}/{num_games} 局游戏")
            
            examples = self.self_play_game()
            all_examples.extend(examples)
            self.total_games += 1
        
        print(f"生成了 {len(all_examples)} 个训练样本")
        return all_examples
    
    def train_network(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """
        训练神经网络
        
        Args:
            examples: 训练样本
            
        Returns:
            训练统计信息
        """
        if len(examples) == 0:
            return {}
        
        # 创建数据集和数据加载器
        dataset = ChessDataset(examples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        self.network.train()
        
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0
        
        for epoch in range(self.training_config['epochs_per_iteration']):
            epoch_loss = 0.0
            
            for batch_boards, batch_policies, batch_values in dataloader:
                batch_boards = batch_boards.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_values = batch_values.to(self.device)
                
                # 前向传播
                pred_policies, pred_values = self.network(batch_boards)
                
                # 计算损失
                policy_loss = -torch.sum(batch_policies * torch.log(pred_policies + 1e-8)) / batch_boards.size(0)
                value_loss = nn.MSELoss()(pred_values.squeeze(), batch_values.squeeze())
                
                total_batch_loss = policy_loss + value_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 统计
                epoch_loss += total_batch_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失
        avg_total_loss = total_loss / (self.training_config['epochs_per_iteration'] * len(dataloader))
        avg_policy_loss = policy_loss_sum / num_batches
        avg_value_loss = value_loss_sum / num_batches
        
        return {
            'total_loss': avg_total_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate_network(self, num_games: int = 20) -> Dict[str, float]:
        """
        评估网络性能
        
        Args:
            num_games: 评估游戏局数
            
        Returns:
            评估结果
        """
        self.network.eval()
        
        wins = 0
        draws = 0
        losses = 0
        
        print(f"开始评估网络性能 ({num_games} 局游戏)...")
        
        for game_idx in range(num_games):
            game_state = GameState()
            move_count = 0
            
            while not game_state.game_over and move_count < 200:
                # 使用贪婪策略
                self.mcts.temperature = 0.0
                best_move = self.mcts.get_best_action(game_state)
                
                success = game_state.make_move(best_move[0], best_move[1])
                if not success:
                    break
                
                move_count += 1
            
            # 统计结果
            winner = game_state.get_winner()
            if winner is None:
                draws += 1
            elif winner == PieceColor.RED:
                wins += 1
            else:
                losses += 1
        
        win_rate = wins / num_games
        draw_rate = draws / num_games
        loss_rate = losses / num_games
        
        return {
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate,
            'games_played': num_games
        }
    
    def train_iteration(self) -> Dict[str, float]:
        """
        执行一次训练迭代
        
        Returns:
            训练统计信息
        """
        self.iteration += 1
        
        print(f"\\n=== 训练迭代 {self.iteration} ===")
        
        # 1. 生成自对弈数据
        new_examples = self.generate_self_play_data(
            self.training_config['self_play_games']
        )
        
        # 2. 添加到缓冲区
        self.training_buffer.extend(new_examples)
        
        # 3. 训练网络
        training_examples = list(self.training_buffer)
        train_stats = self.train_network(training_examples)
        
        # 4. 评估网络
        eval_stats = self.evaluate_network(
            self.training_config['evaluation_games']
        )
        
        # 5. 合并统计信息
        iteration_stats = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'buffer_size': len(self.training_buffer),
            **train_stats,
            **eval_stats
        }
        
        self.training_history.append(iteration_stats)
        
        # 6. 打印统计信息
        print(f"训练损失: {train_stats.get('total_loss', 0):.4f}")
        print(f"策略损失: {train_stats.get('policy_loss', 0):.4f}")
        print(f"价值损失: {train_stats.get('value_loss', 0):.4f}")
        print(f"胜率: {eval_stats.get('win_rate', 0):.3f}")
        print(f"平局率: {eval_stats.get('draw_rate', 0):.3f}")
        print(f"学习率: {train_stats.get('learning_rate', 0):.6f}")
        
        return iteration_stats
    
    def save_checkpoint(self, filepath: str):
        """
        保存检查点
        
        Args:
            filepath: 保存路径
        """
        checkpoint = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'network_config': self.network_config,
            'training_config': self.training_config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        print(f"检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载检查点
        
        Args:
            filepath: 检查点路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint['training_history']
        
        print(f"检查点已从 {filepath} 加载")
    
    def train(self, num_iterations: int, checkpoint_dir: str = "checkpoints"):
        """
        执行完整训练
        
        Args:
            num_iterations: 训练迭代次数
            checkpoint_dir: 检查点保存目录
        """
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"开始训练 {num_iterations} 次迭代")
        print(f"网络参数数量: {self.network.get_model_info()['total_parameters']:,}")
        
        start_time = time.time()
        
        for i in range(num_iterations):
            # 执行训练迭代
            stats = self.train_iteration()
            
            # 保存检查点
            if (self.iteration % self.training_config['checkpoint_interval']) == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_iter_{self.iteration}.pth"
                )
                self.save_checkpoint(checkpoint_path)
        
        total_time = time.time() - start_time
        print(f"\\n训练完成！总用时: {total_time:.2f} 秒")
        
        # 保存最终模型
        final_checkpoint = os.path.join(checkpoint_dir, "final_model.pth")
        self.save_checkpoint(final_checkpoint)


def test_trainer():
    """测试训练器功能"""
    print("测试训练器...")
    
    # 创建训练器（使用较小的配置进行测试）
    network_config = {
        'hidden_channels': 64,
        'num_residual_blocks': 4,
        'use_attention': False
    }
    
    training_config = {
        'mcts_simulations': 50,
        'self_play_games': 2,
        'epochs_per_iteration': 2,
        'batch_size': 8
    }
    
    trainer = ChessTrainer(
        network_config=network_config,
        training_config=training_config
    )
    
    # 测试自对弈
    print("测试自对弈...")
    examples = trainer.generate_self_play_data(1)
    print(f"生成了 {len(examples)} 个训练样本")
    
    # 测试训练
    print("测试网络训练...")
    train_stats = trainer.train_network(examples)
    print(f"训练统计: {train_stats}")
    
    # 测试评估
    print("测试网络评估...")
    eval_stats = trainer.evaluate_network(2)
    print(f"评估统计: {eval_stats}")
    
    print("训练器测试完成！")


if __name__ == "__main__":
    test_trainer()