"""
PPO (Proximal Policy Optimization) 训练器
用于中国象棋AI的强化学习训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import logging
from dataclasses import dataclass

from actor_critic_network import ActorCriticNetwork


@dataclass
class PPOConfig:
    """PPO训练配置"""
    # 基础参数
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    clip_epsilon: float = 0.2  # PPO裁剪参数
    
    # 训练参数
    ppo_epochs: int = 1000  # PPO更新轮数
    mini_batch_size: int = 64  # 小批次大小
    max_grad_norm: float = 0.5  # 梯度裁剪
    
    # 损失函数权重
    value_loss_coef: float = 0.5  # 价值损失系数
    entropy_coef: float = 0.01  # 熵损失系数
    
    # 其他参数
    normalize_advantages: bool = True  # 是否标准化优势
    use_clipped_value_loss: bool = True  # 是否使用裁剪价值损失
    target_kl: float = 0.01  # 目标KL散度（早停）


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, buffer_size: int, observation_shape: Tuple[int, ...], action_space_size: int):
        """
        初始化缓冲区
        
        Args:
            buffer_size: 缓冲区大小
            observation_shape: 观测空间形状
            action_space_size: 动作空间大小
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_space_size = action_space_size
        
        # 初始化缓冲区
        self.observations = np.zeros((buffer_size,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.action_masks = np.zeros((buffer_size, action_space_size), dtype=np.float32)
        
        # 计算的优势和回报
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def add(self, obs: np.ndarray, action: int, reward: float, value: float, 
            log_prob: float, done: bool, action_mask: np.ndarray):
        """添加经验到缓冲区"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.action_masks[self.ptr] = action_mask
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
    
    def compute_advantages_and_returns(self, last_value: float, gamma: float, gae_lambda: float):
        """计算优势函数和回报"""
        size = self.buffer_size if self.full else self.ptr
        
        # 计算GAE优势
        advantages = np.zeros_like(self.rewards[:size])
        last_gae_lam = 0
        
        for step in reversed(range(size)):
            if step == size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        # 计算回报
        returns = advantages + self.values[:size]
        
        self.advantages[:size] = advantages
        self.returns[:size] = returns
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """获取训练批次"""
        size = self.buffer_size if self.full else self.ptr
        indices = np.random.choice(size, batch_size, replace=False)
        
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'old_log_probs': torch.FloatTensor(self.log_probs[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices]),
            'old_values': torch.FloatTensor(self.values[indices]),
            'action_masks': torch.FloatTensor(self.action_masks[indices])
        }
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.full = False


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, 
                 network: ActorCriticNetwork,
                 config: PPOConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化PPO训练器
        
        Args:
            network: Actor-Critic网络
            config: PPO配置
            device: 计算设备
        """
        self.network = network.to(device)
        self.config = config or PPOConfig()
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # 训练统计
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'clip_fraction': deque(maxlen=100),
            'explained_variance': deque(maxlen=100)
        }
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def compute_policy_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算策略损失"""
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        action_masks = batch['action_masks'].to(self.device)
        
        # 标准化优势
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 获取当前策略的log概率和熵
        log_probs, _, entropy = self.network.evaluate_actions(observations, actions, action_masks)
        
        # 计算比率
        ratio = torch.exp(log_probs - old_log_probs)
        
        # 计算PPO损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算统计信息
        with torch.no_grad():
            kl_div = (old_log_probs - log_probs).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
        
        stats = {
            'kl_divergence': kl_div,
            'clip_fraction': clip_fraction,
            'entropy': entropy.mean().item()
        }
        
        return policy_loss, entropy.mean(), stats
    
    def compute_value_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算价值损失"""
        observations = batch['observations'].to(self.device)
        returns = batch['returns'].to(self.device)
        old_values = batch['old_values'].to(self.device)
        
        # 获取当前价值预测
        _, _, values = self.network(observations)
        values = values.squeeze(-1)
        
        if self.config.use_clipped_value_loss:
            # 裁剪价值损失
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.config.clip_epsilon, self.config.clip_epsilon
            )
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_pred_clipped - returns).pow(2)
            value_loss = torch.max(value_loss1, value_loss2).mean()
        else:
            # 标准MSE损失
            value_loss = nn.MSELoss()(values, returns)
        
        return value_loss
    
    def compute_explained_variance(self, batch: Dict[str, torch.Tensor]) -> float:
        """计算解释方差"""
        observations = batch['observations'].to(self.device)
        returns = batch['returns'].to(self.device)
        
        with torch.no_grad():
            _, _, values = self.network(observations)
            values = values.squeeze(-1)
            
            var_y = torch.var(returns)
            explained_var = 1 - torch.var(returns - values) / (var_y + 1e-8)
            
        return explained_var.item()
    
    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """执行PPO更新"""
        update_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': []
        }
        
        # 多轮更新
        for epoch in range(self.config.ppo_epochs):
            # 获取小批次数据
            batch = rollout_buffer.get_batch(self.config.mini_batch_size)
            
            # 计算损失
            policy_loss, entropy, policy_stats = self.compute_policy_loss(batch)
            value_loss = self.compute_value_loss(batch)
            explained_var = self.compute_explained_variance(batch)
            
            # 总损失
            total_loss = (policy_loss + 
                         self.config.value_loss_coef * value_loss - 
                         self.config.entropy_coef * entropy)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            # 记录统计信息
            update_stats['policy_loss'].append(policy_loss.item())
            update_stats['value_loss'].append(value_loss.item())
            update_stats['entropy_loss'].append(entropy.item())
            update_stats['total_loss'].append(total_loss.item())
            update_stats['kl_divergence'].append(policy_stats['kl_divergence'])
            update_stats['clip_fraction'].append(policy_stats['clip_fraction'])
            update_stats['explained_variance'].append(explained_var)
            
            # 早停检查
            if policy_stats['kl_divergence'] > self.config.target_kl:
                self.logger.info(f"Early stopping at epoch {epoch} due to KL divergence: {policy_stats['kl_divergence']:.4f}")
                break
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均统计信息
        avg_stats = {key: np.mean(values) for key, values in update_stats.items()}
        
        # 更新训练统计
        for key, value in avg_stats.items():
            self.training_stats[key].append(value)
        
        return avg_stats
    
    def get_training_stats(self) -> Dict[str, float]:
        """获取训练统计信息"""
        return {
            key: np.mean(values) if values else 0.0 
            for key, values in self.training_stats.items()
        }
    
    def save_checkpoint(self, filepath: str, episode: int, total_steps: int):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'total_steps': total_steps,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_stats': dict(self.training_stats)
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Tuple[int, int]:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练统计
        for key, values in checkpoint['training_stats'].items():
            self.training_stats[key] = deque(values, maxlen=100)
        
        episode = checkpoint['episode']
        total_steps = checkpoint['total_steps']
        
        self.logger.info(f"Checkpoint loaded from {filepath}, episode: {episode}, steps: {total_steps}")
        
        return episode, total_steps


class AdvantageCalculator:
    """优势函数计算器"""
    
    @staticmethod
    def compute_gae(rewards: np.ndarray, 
                   values: np.ndarray, 
                   dones: np.ndarray,
                   gamma: float = 0.99, 
                   gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE优势函数
        
        Args:
            rewards: 奖励序列
            values: 价值预测序列
            dones: 结束标志序列
            gamma: 折扣因子
            gae_lambda: GAE参数
            
        Returns:
            advantages: 优势函数
            returns: 回报
        """
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = 0  # 假设最后一步的下一个价值为0
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            
            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        
        return advantages, returns
    
    @staticmethod
    def compute_td_lambda(rewards: np.ndarray,
                         values: np.ndarray,
                         dones: np.ndarray,
                         gamma: float = 0.99,
                         lambda_: float = 0.95) -> np.ndarray:
        """
        计算TD(λ)回报
        
        Args:
            rewards: 奖励序列
            values: 价值预测序列
            dones: 结束标志序列
            gamma: 折扣因子
            lambda_: λ参数
            
        Returns:
            td_lambda_returns: TD(λ)回报
        """
        returns = np.zeros_like(rewards)
        
        for t in range(len(rewards)):
            returns[t] = rewards[t]
            
            # 计算λ-回报
            lambda_return = 0
            for k in range(1, len(rewards) - t):
                if dones[t + k - 1]:
                    break
                
                # n步回报
                n_step_return = sum([
                    (gamma ** i) * rewards[t + i] 
                    for i in range(k)
                ]) + (gamma ** k) * values[t + k]
                
                # 加权
                weight = (lambda_ ** (k - 1)) * (1 - lambda_) if k < len(rewards) - t else (lambda_ ** (k - 1))
                lambda_return += weight * n_step_return
            
            returns[t] += lambda_return
        
        return returns


def test_ppo_trainer():
    """测试PPO训练器"""
    print("Testing PPO Trainer...")
    
    # 创建网络和训练器
    from actor_critic_network import create_actor_critic_network
    
    network = create_actor_critic_network()
    trainer = PPOTrainer(network)
    
    # 创建缓冲区
    buffer = RolloutBuffer(
        buffer_size=1000,
        observation_shape=(14, 10, 9),
        action_space_size=2086
    )
    
    # 添加一些虚拟数据
    for i in range(100):
        obs = np.random.randn(14, 10, 9)
        action = np.random.randint(0, 2086)
        reward = np.random.randn()
        value = np.random.randn()
        log_prob = np.random.randn()
        done = i % 20 == 19  # 每20步结束一次
        action_mask = np.ones(2086)
        
        buffer.add(obs, action, reward, value, log_prob, done, action_mask)
    
    # 计算优势和回报
    buffer.compute_advantages_and_returns(0.0, 0.99, 0.95)
    
    # 执行更新
    stats = trainer.update(buffer)
    
    print("Update stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("PPO Trainer test completed successfully!")


if __name__ == "__main__":
    test_ppo_trainer()