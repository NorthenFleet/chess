"""
MCTS + PPO 训练器
整合蒙特卡洛树搜索和PPO算法的训练流程
"""

import torch
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
import os
from collections import defaultdict

from actor_critic_network import ActorCriticNetwork, create_actor_critic_network
from ppo_trainer import PPOTrainer, PPOConfig, RolloutBuffer
from mcts import MCTS
from board import ChessBoard
from encoder import ChessBoardEncoder


@dataclass
class MCTSPPOConfig:
    """MCTS+PPO训练配置"""
    # MCTS参数
    mcts_simulations: int = 800  # MCTS模拟次数
    mcts_c_puct: float = 1.0  # MCTS探索常数
    mcts_temperature: float = 1.0  # MCTS温度参数
    mcts_dirichlet_alpha: float = 0.3  # 狄利克雷噪声参数
    mcts_dirichlet_epsilon: float = 0.25  # 狄利克雷噪声权重
    
    # PPO参数
    ppo_config: PPOConfig = None
    
    # 训练参数
    episodes_per_iteration: int = 100  # 每次迭代的对局数
    buffer_size: int = 10000  # 经验缓冲区大小
    min_buffer_size: int = 1000  # 最小缓冲区大小（开始训练）
    
    # 自对弈参数
    self_play_games: int = 50  # 每次迭代的自对弈局数
    evaluation_games: int = 20  # 评估局数
    evaluation_interval: int = 10  # 评估间隔（迭代次数）
    
    # 保存参数
    save_interval: int = 50  # 保存间隔（迭代次数）
    checkpoint_dir: str = "checkpoints"  # 检查点目录
    
    # 其他参数
    max_game_length: int = 200  # 最大游戏长度
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class GameResult:
    """游戏结果"""
    
    def __init__(self):
        self.states = []  # 游戏状态序列
        self.actions = []  # 动作序列
        self.mcts_probs = []  # MCTS概率分布
        self.rewards = []  # 奖励序列
        self.winner = None  # 获胜者
        self.game_length = 0  # 游戏长度


class MCTSPPOTrainer:
    """MCTS+PPO训练器"""
    
    def __init__(self, config: MCTSPPOConfig = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config or MCTSPPOConfig()
        if self.config.ppo_config is None:
            self.config.ppo_config = PPOConfig()
        
        # 设备
        self.device = torch.device(self.config.device)
        
        # 创建网络
        self.network = create_actor_critic_network().to(self.device)
        
        # 创建PPO训练器
        self.ppo_trainer = PPOTrainer(self.network, self.config.ppo_config, self.device)
        
        # 创建编码器
        self.encoder = ChessBoardEncoder()
        
        # 创建经验缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=self.config.buffer_size,
            observation_shape=(14, 10, 9),
            action_space_size=2086
        )
        
        # 训练统计
        self.training_stats = {
            'iteration': 0,
            'total_games': 0,
            'win_rate': 0.0,
            'avg_game_length': 0.0,
            'avg_reward': 0.0,
            'mcts_policy_entropy': 0.0,
            'network_policy_entropy': 0.0
        }
        
        # 创建检查点目录
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # 日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def create_mcts(self, network: ActorCriticNetwork) -> MCTS:
        """创建MCTS实例"""
        return MCTS(
            network=network,
            encoder=self.encoder,
            c_puct=self.config.mcts_c_puct,
            num_simulations=self.config.mcts_simulations,
            device=self.device
        )
    
    def play_game(self, mcts1: MCTS, mcts2: MCTS = None, 
                  temperature: float = 1.0, add_noise: bool = True) -> GameResult:
        """
        进行一局游戏
        
        Args:
            mcts1: 红方MCTS
            mcts2: 黑方MCTS（如果为None，则使用mcts1）
            temperature: 温度参数
            add_noise: 是否添加狄利克雷噪声
            
        Returns:
            GameResult: 游戏结果
        """
        if mcts2 is None:
            mcts2 = mcts1
        
        board = ChessBoard()
        result = GameResult()
        
        current_mcts = mcts1
        move_count = 0
        
        while not board.is_game_over() and move_count < self.config.max_game_length:
            # 编码当前状态
            state = self.encoder.encode_board(board)
            legal_moves = board.get_legal_moves()
            action_mask = self.encoder.create_action_mask(legal_moves)
            
            # MCTS搜索
            if add_noise and move_count < 30:  # 前30步添加噪声
                mcts_probs = current_mcts.search_with_noise(
                    board, 
                    alpha=self.config.mcts_dirichlet_alpha,
                    epsilon=self.config.mcts_dirichlet_epsilon
                )
            else:
                mcts_probs = current_mcts.search(board)
            
            # 根据温度参数选择动作
            if temperature > 0:
                # 温度采样
                probs = np.power(mcts_probs, 1.0 / temperature)
                probs = probs / np.sum(probs)
                action_idx = np.random.choice(len(probs), p=probs)
            else:
                # 贪婪选择
                action_idx = np.argmax(mcts_probs)
            
            action = legal_moves[action_idx]
            
            # 记录状态和动作
            result.states.append(state.copy())
            result.actions.append(self.encoder.encode_action(action))
            result.mcts_probs.append(mcts_probs.copy())
            
            # 执行动作
            board.make_move(action)
            move_count += 1
            
            # 切换玩家
            current_mcts = mcts2 if current_mcts == mcts1 else mcts1
        
        # 计算游戏结果
        if board.is_checkmate():
            winner = 1 - board.current_player  # 当前玩家被将死，对手获胜
            result.winner = winner
        elif board.is_stalemate() or move_count >= self.config.max_game_length:
            result.winner = -1  # 平局
        else:
            result.winner = -1  # 其他情况视为平局
        
        result.game_length = move_count
        
        # 计算奖励
        result.rewards = self._compute_rewards(result)
        
        return result
    
    def _compute_rewards(self, result: GameResult) -> List[float]:
        """
        计算奖励序列
        
        Args:
            result: 游戏结果
            
        Returns:
            奖励序列
        """
        rewards = []
        game_length = len(result.states)
        
        for i in range(game_length):
            player = i % 2  # 0为红方，1为黑方
            
            # 基础奖励
            if result.winner == player:
                # 获胜奖励，越早获胜奖励越高
                base_reward = 1.0 + (self.config.max_game_length - result.game_length) / self.config.max_game_length
            elif result.winner == 1 - player:
                # 失败惩罚
                base_reward = -1.0
            else:
                # 平局
                base_reward = 0.0
            
            # 步骤奖励（鼓励快速决策）
            step_reward = -0.01
            
            # 总奖励
            total_reward = base_reward + step_reward
            rewards.append(total_reward)
        
        return rewards
    
    def collect_self_play_data(self, num_games: int) -> List[GameResult]:
        """
        收集自对弈数据
        
        Args:
            num_games: 游戏数量
            
        Returns:
            游戏结果列表
        """
        self.logger.info(f"Collecting {num_games} self-play games...")
        
        results = []
        mcts = self.create_mcts(self.network)
        
        for game_idx in range(num_games):
            # 前30%的游戏使用高温度，后面逐渐降低
            if game_idx < num_games * 0.3:
                temperature = self.config.mcts_temperature
            elif game_idx < num_games * 0.7:
                temperature = self.config.mcts_temperature * 0.5
            else:
                temperature = 0.1
            
            result = self.play_game(mcts, temperature=temperature, add_noise=True)
            results.append(result)
            
            if (game_idx + 1) % 10 == 0:
                self.logger.info(f"Completed {game_idx + 1}/{num_games} games")
        
        return results
    
    def add_game_to_buffer(self, result: GameResult):
        """将游戏结果添加到缓冲区"""
        for i, (state, action, reward) in enumerate(zip(result.states, result.actions, result.rewards)):
            # 获取网络预测的价值
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, _, value = self.network(state_tensor)
                value = value.item()
            
            # 计算动作概率和log概率
            with torch.no_grad():
                legal_moves = np.where(result.mcts_probs[i] > 0)[0]
                action_mask = np.zeros(2086)
                action_mask[legal_moves] = 1.0
                
                action_probs, _, _ = self.network(state_tensor, torch.FloatTensor(action_mask).unsqueeze(0).to(self.device))
                action_prob = action_probs[0, action].item()
                log_prob = np.log(action_prob + 1e-8)
            
            # 游戏是否结束
            done = (i == len(result.states) - 1)
            
            # 添加到缓冲区
            self.buffer.add(state, action, reward, value, log_prob, done, action_mask)
    
    def train_iteration(self):
        """执行一次训练迭代"""
        self.logger.info(f"Starting training iteration {self.training_stats['iteration'] + 1}")
        
        # 收集自对弈数据
        game_results = self.collect_self_play_data(self.config.self_play_games)
        
        # 添加数据到缓冲区
        for result in game_results:
            self.add_game_to_buffer(result)
        
        # 更新统计信息
        self.training_stats['total_games'] += len(game_results)
        avg_game_length = np.mean([r.game_length for r in game_results])
        avg_reward = np.mean([np.mean(r.rewards) for r in game_results])
        win_rate = np.mean([1 if r.winner == 0 else 0 for r in game_results])  # 红方胜率
        
        self.training_stats['avg_game_length'] = avg_game_length
        self.training_stats['avg_reward'] = avg_reward
        self.training_stats['win_rate'] = win_rate
        
        # 如果缓冲区有足够数据，进行PPO训练
        if self.buffer.ptr >= self.config.min_buffer_size or self.buffer.full:
            self.logger.info("Training PPO...")
            
            # 计算优势和回报
            last_value = 0.0  # 假设最后状态的价值为0
            self.buffer.compute_advantages_and_returns(
                last_value, 
                self.config.ppo_config.gamma, 
                self.config.ppo_config.gae_lambda
            )
            
            # PPO更新
            ppo_stats = self.ppo_trainer.update(self.buffer)
            
            # 清空缓冲区
            self.buffer.clear()
            
            self.logger.info(f"PPO training completed. Stats: {ppo_stats}")
        
        # 更新迭代计数
        self.training_stats['iteration'] += 1
        
        # 记录训练信息
        self.logger.info(f"Iteration {self.training_stats['iteration']} completed:")
        self.logger.info(f"  Games played: {len(game_results)}")
        self.logger.info(f"  Average game length: {avg_game_length:.2f}")
        self.logger.info(f"  Average reward: {avg_reward:.4f}")
        self.logger.info(f"  Win rate: {win_rate:.4f}")
    
    def evaluate_network(self, num_games: int = None) -> Dict[str, float]:
        """
        评估网络性能
        
        Args:
            num_games: 评估游戏数量
            
        Returns:
            评估结果
        """
        if num_games is None:
            num_games = self.config.evaluation_games
        
        self.logger.info(f"Evaluating network with {num_games} games...")
        
        # 创建两个MCTS实例（当前网络 vs 当前网络）
        mcts1 = self.create_mcts(self.network)
        mcts2 = self.create_mcts(self.network)
        
        results = []
        for _ in range(num_games):
            result = self.play_game(mcts1, mcts2, temperature=0.1, add_noise=False)
            results.append(result)
        
        # 计算评估指标
        win_rate = np.mean([1 if r.winner == 0 else 0 for r in results])
        avg_game_length = np.mean([r.game_length for r in results])
        avg_reward = np.mean([np.mean(r.rewards) for r in results])
        
        eval_stats = {
            'win_rate': win_rate,
            'avg_game_length': avg_game_length,
            'avg_reward': avg_reward,
            'games_played': num_games
        }
        
        self.logger.info(f"Evaluation completed: {eval_stats}")
        
        return eval_stats
    
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_iter_{self.training_stats['iteration']}.pt"
        )
        
        self.ppo_trainer.save_checkpoint(
            checkpoint_path,
            self.training_stats['iteration'],
            self.training_stats['total_games']
        )
        
        # 保存训练统计
        stats_path = os.path.join(
            self.config.checkpoint_dir,
            f"training_stats_iter_{self.training_stats['iteration']}.pt"
        )
        torch.save(self.training_stats, stats_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        iteration, total_games = self.ppo_trainer.load_checkpoint(checkpoint_path)
        
        # 更新训练统计
        self.training_stats['iteration'] = iteration
        self.training_stats['total_games'] = total_games
        
        # 尝试加载训练统计
        stats_path = checkpoint_path.replace('checkpoint_', 'training_stats_')
        if os.path.exists(stats_path):
            self.training_stats.update(torch.load(stats_path))
    
    def train(self, num_iterations: int, resume_from: str = None):
        """
        开始训练
        
        Args:
            num_iterations: 训练迭代次数
            resume_from: 恢复训练的检查点路径
        """
        if resume_from:
            self.load_checkpoint(resume_from)
            self.logger.info(f"Resumed training from iteration {self.training_stats['iteration']}")
        
        start_iteration = self.training_stats['iteration']
        
        for iteration in range(start_iteration, start_iteration + num_iterations):
            start_time = time.time()
            
            # 执行训练迭代
            self.train_iteration()
            
            # 评估网络
            if iteration % self.config.evaluation_interval == 0:
                eval_stats = self.evaluate_network()
                self.training_stats.update(eval_stats)
            
            # 保存检查点
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint()
            
            iteration_time = time.time() - start_time
            self.logger.info(f"Iteration {iteration + 1} completed in {iteration_time:.2f}s")
        
        # 最终保存
        self.save_checkpoint()
        self.logger.info("Training completed!")


def main():
    """主函数"""
    # 配置
    config = MCTSPPOConfig(
        mcts_simulations=400,  # 减少模拟次数以加快训练
        self_play_games=20,    # 减少自对弈局数
        episodes_per_iteration=50,
        evaluation_interval=5,
        save_interval=10
    )
    
    # 创建训练器
    trainer = MCTSPPOTrainer(config)
    
    # 开始训练
    trainer.train(num_iterations=100)


if __name__ == "__main__":
    main()