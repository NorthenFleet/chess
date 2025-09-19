"""
Actor-Critic网络架构
用于中国象棋AI的PPO训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int = 256):
        """
        初始化残差块
        
        Args:
            channels: 通道数
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, channels: int = 256, num_heads: int = 8):
        """
        初始化自注意力层
        
        Args:
            channels: 输入通道数
            num_heads: 注意力头数
        """
        super(SelfAttention, self).__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, channels, height, width = x.shape
        
        # 计算Q, K, V
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # 计算注意力权重
        attention_weights = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 应用注意力
        attended = torch.matmul(v, attention_weights.transpose(-2, -1))
        attended = attended.view(batch_size, channels, height, width)
        
        # 输出投影
        out = self.out_proj(attended)
        
        # 残差连接和层归一化
        out = out + x
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return out


class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络"""
    
    def __init__(self, 
                 input_channels: int = 14,
                 hidden_channels: int = 256,
                 num_residual_blocks: int = 20,
                 num_attention_heads: int = 8,
                 action_space_size: int = 2086,
                 use_attention: bool = True):
        """
        初始化Actor网络
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏层通道数
            num_residual_blocks: 残差块数量
            num_attention_heads: 注意力头数
            action_space_size: 动作空间大小
            use_attention: 是否使用注意力机制
        """
        super(ActorNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_residual_blocks = num_residual_blocks
        self.use_attention = use_attention
        
        # 初始卷积层
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, 
                                   kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(hidden_channels)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
        ])
        
        # 自注意力机制
        if use_attention:
            self.attention = SelfAttention(hidden_channels, num_attention_heads)
        
        # 策略头
        self.policy_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 10 * 9, action_space_size)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入棋盘状态，形状为 (batch_size, input_channels, 10, 9)
            action_mask: 动作掩码，形状为 (batch_size, action_space_size)
            
        Returns:
            动作概率分布，形状为 (batch_size, action_space_size)
        """
        # 初始卷积
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)
        
        # 残差块
        for block in self.residual_blocks:
            out = block(out)
        
        # 注意力机制
        if self.use_attention:
            out = self.attention(out)
        
        # 策略头
        policy_out = self.policy_conv(out)
        policy_out = self.policy_bn(policy_out)
        policy_out = F.relu(policy_out)
        
        # 展平并通过全连接层
        policy_out = policy_out.view(policy_out.size(0), -1)
        policy_logits = self.policy_fc(policy_out)
        
        # 应用动作掩码
        if action_mask is not None:
            policy_logits = policy_logits + (action_mask - 1) * 1e9
        
        # 返回动作概率分布
        action_probs = F.softmax(policy_logits, dim=1)
        
        return action_probs, policy_logits


class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络"""
    
    def __init__(self, 
                 input_channels: int = 14,
                 hidden_channels: int = 256,
                 num_residual_blocks: int = 20,
                 num_attention_heads: int = 8,
                 use_attention: bool = True):
        """
        初始化Critic网络
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏层通道数
            num_residual_blocks: 残差块数量
            num_attention_heads: 注意力头数
            use_attention: 是否使用注意力机制
        """
        super(CriticNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_residual_blocks = num_residual_blocks
        self.use_attention = use_attention
        
        # 初始卷积层
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, 
                                   kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(hidden_channels)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
        ])
        
        # 自注意力机制
        if use_attention:
            self.attention = SelfAttention(hidden_channels, num_attention_heads)
        
        # 价值头
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入棋盘状态，形状为 (batch_size, input_channels, 10, 9)
            
        Returns:
            状态价值，形状为 (batch_size, 1)
        """
        # 初始卷积
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)
        
        # 残差块
        for block in self.residual_blocks:
            out = block(out)
        
        # 注意力机制
        if self.use_attention:
            out = self.attention(out)
        
        # 价值头
        value_out = self.value_conv(out)
        value_out = self.value_bn(value_out)
        value_out = F.relu(value_out)
        
        # 展平并通过全连接层
        value_out = value_out.view(value_out.size(0), -1)
        value_out = self.value_fc1(value_out)
        value_out = F.relu(value_out)
        value = self.value_fc2(value_out)
        
        # Tanh激活，输出范围 [-1, 1]
        value = torch.tanh(value)
        
        return value


class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络架构"""
    
    def __init__(self, 
                 input_channels: int = 14,
                 hidden_channels: int = 256,
                 num_residual_blocks: int = 20,
                 num_attention_heads: int = 8,
                 action_space_size: int = 2086,
                 use_attention: bool = True,
                 shared_backbone: bool = False):
        """
        初始化Actor-Critic网络
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏层通道数
            num_residual_blocks: 残差块数量
            num_attention_heads: 注意力头数
            action_space_size: 动作空间大小
            use_attention: 是否使用注意力机制
            shared_backbone: 是否共享主干网络
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.shared_backbone = shared_backbone
        
        if shared_backbone:
            # 共享主干网络
            self.backbone = self._create_backbone(
                input_channels, hidden_channels, num_residual_blocks, 
                num_attention_heads, use_attention
            )
            
            # 分离的头部
            self.actor_head = self._create_actor_head(hidden_channels, action_space_size)
            self.critic_head = self._create_critic_head(hidden_channels)
        else:
            # 独立的Actor和Critic网络
            self.actor = ActorNetwork(
                input_channels, hidden_channels, num_residual_blocks,
                num_attention_heads, action_space_size, use_attention
            )
            self.critic = CriticNetwork(
                input_channels, hidden_channels, num_residual_blocks,
                num_attention_heads, use_attention
            )
    
    def _create_backbone(self, input_channels, hidden_channels, num_residual_blocks, 
                        num_attention_heads, use_attention):
        """创建共享主干网络"""
        layers = []
        
        # 初始卷积层
        layers.extend([
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        ])
        
        # 残差块
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_channels))
        
        # 注意力机制
        if use_attention:
            layers.append(SelfAttention(hidden_channels, num_attention_heads))
        
        return nn.Sequential(*layers)
    
    def _create_actor_head(self, hidden_channels, action_space_size):
        """创建Actor头部"""
        return nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 10 * 9, action_space_size)
        )
    
    def _create_critic_head(self, hidden_channels):
        """创建Critic头部"""
        return nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(10 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入棋盘状态，形状为 (batch_size, input_channels, 10, 9)
            action_mask: 动作掩码，形状为 (batch_size, action_space_size)
            
        Returns:
            action_probs: 动作概率分布，形状为 (batch_size, action_space_size)
            action_logits: 动作logits，形状为 (batch_size, action_space_size)
            state_value: 状态价值，形状为 (batch_size, 1)
        """
        if self.shared_backbone:
            # 共享主干网络
            features = self.backbone(x)
            
            # Actor头部
            actor_logits = self.actor_head(features)
            if action_mask is not None:
                actor_logits = actor_logits + (action_mask - 1) * 1e9
            action_probs = F.softmax(actor_logits, dim=1)
            
            # Critic头部
            state_value = self.critic_head(features)
            
        else:
            # 独立网络
            action_probs, actor_logits = self.actor(x, action_mask)
            state_value = self.critic(x)
        
        return action_probs, actor_logits, state_value
    
    def get_action_and_value(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作和价值（用于PPO训练）
        
        Args:
            x: 输入棋盘状态
            action_mask: 动作掩码
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
            state_value: 状态价值
        """
        action_probs, action_logits, state_value = self.forward(x, action_mask)
        
        # 从概率分布中采样动作
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor, 
                        action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作（用于PPO训练）
        
        Args:
            x: 输入棋盘状态
            actions: 要评估的动作
            action_mask: 动作掩码
            
        Returns:
            log_probs: 动作的对数概率
            state_values: 状态价值
            entropy: 策略熵
        """
        action_probs, action_logits, state_values = self.forward(x, action_mask)
        
        # 计算动作的对数概率
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # 计算策略熵
        entropy = dist.entropy()
        
        return log_probs, state_values, entropy


def create_actor_critic_network(config: dict = None) -> ActorCriticNetwork:
    """
    创建Actor-Critic网络
    
    Args:
        config: 网络配置字典
        
    Returns:
        ActorCriticNetwork实例
    """
    if config is None:
        config = {}
    
    default_config = {
        'input_channels': 14,
        'hidden_channels': 256,
        'num_residual_blocks': 20,
        'num_attention_heads': 8,
        'action_space_size': 2086,
        'use_attention': True,
        'shared_backbone': False
    }
    
    # 合并配置
    default_config.update(config)
    
    return ActorCriticNetwork(**default_config)


def test_actor_critic_network():
    """测试Actor-Critic网络"""
    print("Testing Actor-Critic Network...")
    
    # 创建网络
    network = create_actor_critic_network()
    
    # 创建测试输入
    batch_size = 4
    input_tensor = torch.randn(batch_size, 14, 10, 9)
    action_mask = torch.ones(batch_size, 2086)
    
    # 前向传播
    action_probs, action_logits, state_values = network(input_tensor, action_mask)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Action probs shape: {action_probs.shape}")
    print(f"Action logits shape: {action_logits.shape}")
    print(f"State values shape: {state_values.shape}")
    
    # 测试动作采样
    actions, log_probs, values = network.get_action_and_value(input_tensor, action_mask)
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Values shape: {values.shape}")
    
    # 测试动作评估
    eval_log_probs, eval_values, entropy = network.evaluate_actions(input_tensor, actions, action_mask)
    print(f"Eval log probs shape: {eval_log_probs.shape}")
    print(f"Eval values shape: {eval_values.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    print("Actor-Critic Network test completed successfully!")


if __name__ == "__main__":
    test_actor_critic_network()