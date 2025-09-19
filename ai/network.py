"""
中国象棋神经网络模型

基于ResNet + 注意力机制的双头网络架构
包含策略网络和价值网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


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
        
        # 残差连接
        out += residual
        out = F.relu(out)
        
        return out


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, channels: int = 256, num_heads: int = 8):
        """
        初始化自注意力模块
        
        Args:
            channels: 输入通道数
            num_heads: 注意力头数
        """
        super(SelfAttention, self).__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels必须能被num_heads整除"
        
        # Query, Key, Value投影层
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        # 输出投影层
        self.out_proj = nn.Linear(channels, channels)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, height, width)
            
        Returns:
            输出张量，形状与输入相同
        """
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        
        # 重塑为序列格式: (batch_size, seq_len, channels)
        x_flat = x.view(batch_size, channels, seq_len).transpose(1, 2)
        
        # 计算 Q, K, V
        Q = self.query(x_flat)  # (batch_size, seq_len, channels)
        K = self.key(x_flat)
        V = self.value(x_flat)
        
        # 重塑为多头格式: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)
        
        # 合并多头: (batch_size, seq_len, channels)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        
        # 输出投影
        output = self.out_proj(attended)
        
        # 残差连接
        output = output + x_flat
        
        # 重塑回原始格式: (batch_size, channels, height, width)
        output = output.transpose(1, 2).view(batch_size, channels, height, width)
        
        return output


class PolicyHead(nn.Module):
    """策略头：输出动作概率分布"""
    
    def __init__(self, input_channels: int = 256, action_space_size: int = 8100):
        """
        初始化策略头
        
        Args:
            input_channels: 输入通道数
            action_space_size: 动作空间大小
        """
        super(PolicyHead, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 10 * 9, action_space_size)  # 2 * 10 * 9 = 180
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 (batch_size, input_channels, 10, 9)
            
        Returns:
            策略概率分布，形状为 (batch_size, action_space_size)
        """
        # 卷积层
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = self.fc(out)
        
        # Softmax得到概率分布
        policy_probs = F.softmax(out, dim=1)
        
        return policy_probs


class ValueHead(nn.Module):
    """价值网络头"""
    
    def __init__(self, input_channels: int = 256):
        """
        初始化价值头
        
        Args:
            input_channels: 输入通道数
        """
        super(ValueHead, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(10 * 9, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 (batch_size, input_channels, 10, 9)
            
        Returns:
            价值评估，形状为 (batch_size, 1)，范围 [-1, 1]
        """
        # 卷积层
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        
        # Tanh激活，输出范围 [-1, 1]
        value = torch.tanh(out)
        
        return value


class ChessNet(nn.Module):
    """中国象棋神经网络"""
    
    def __init__(self, 
                 input_channels: int = 14,
                 hidden_channels: int = 256,
                 num_residual_blocks: int = 20,
                 num_attention_heads: int = 8,
                 action_space_size: int = 8100,
                 use_attention: bool = True):
        """
        初始化神经网络
        
        Args:
            input_channels: 输入通道数 (14个棋子类型通道)
            hidden_channels: 隐藏层通道数
            num_residual_blocks: 残差块数量
            num_attention_heads: 注意力头数
            action_space_size: 动作空间大小
            use_attention: 是否使用注意力机制
        """
        super(ChessNet, self).__init__()
        
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
        
        # 策略头和价值头
        self.policy_head = PolicyHead(hidden_channels, action_space_size)
        self.value_head = ValueHead(hidden_channels)
        
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
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入棋盘状态，形状为 (batch_size, 14, 10, 9)
            action_mask: 合法动作掩码，形状为 (batch_size, action_space_size)
            
        Returns:
            (策略概率分布, 价值评估)
        """
        # 初始卷积
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)
        
        # 残差块
        for residual_block in self.residual_blocks:
            out = residual_block(out)
        
        # 注意力机制
        if self.use_attention:
            out = self.attention(out)
        
        # 策略头
        policy_logits = self.policy_head(out)
        
        # 应用动作掩码（如果提供）
        if action_mask is not None:
            # 将非法动作的概率设为极小值
            policy_logits = policy_logits * action_mask + (1 - action_mask) * (-1e8)
            policy_probs = F.softmax(policy_logits, dim=1)
        else:
            policy_probs = policy_logits
        
        # 价值头
        value = self.value_head(out)
        
        return policy_probs, value
    
    def predict(self, board_state: np.ndarray, 
                action_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        单次预测
        
        Args:
            board_state: 棋盘状态，形状为 (14, 10, 9)
            action_mask: 合法动作掩码，形状为 (action_space_size,)
            
        Returns:
            (策略概率分布, 价值评估)
        """
        self.eval()
        
        with torch.no_grad():
            # 转换为张量并添加batch维度
            x = torch.FloatTensor(board_state).unsqueeze(0)
            
            if action_mask is not None:
                mask = torch.FloatTensor(action_mask).unsqueeze(0)
            else:
                mask = None
            
            # 前向传播
            policy_probs, value = self.forward(x, mask)
            
            # 转换回numpy
            policy_probs = policy_probs.squeeze(0).numpy()
            value = value.squeeze(0).item()
            
        return policy_probs, value
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_channels': self.input_channels,
            'hidden_channels': self.hidden_channels,
            'num_residual_blocks': self.num_residual_blocks,
            'use_attention': self.use_attention,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }


def create_chess_net(config: dict = None) -> ChessNet:
    """
    创建象棋神经网络
    
    Args:
        config: 配置字典
        
    Returns:
        ChessNet实例
    """
    default_config = {
        'input_channels': 14,
        'hidden_channels': 256,
        'num_residual_blocks': 20,
        'num_attention_heads': 8,
        'action_space_size': 8100,
        'use_attention': True
    }
    
    if config:
        default_config.update(config)
    
    return ChessNet(**default_config)


def test_network():
    """测试网络功能"""
    print("测试中国象棋神经网络...")
    
    # 创建网络
    net = create_chess_net()
    
    # 打印模型信息
    info = net.get_model_info()
    print(f"模型参数数量: {info['total_parameters']:,}")
    print(f"模型大小: {info['model_size_mb']:.2f} MB")
    
    # 创建测试输入
    batch_size = 4
    x = torch.randn(batch_size, 14, 10, 9)
    action_mask = torch.ones(batch_size, 8100)  # 所有动作都合法
    
    # 前向传播
    net.eval()
    with torch.no_grad():
        policy_probs, value = net(x, action_mask)
    
    print(f"策略输出形状: {policy_probs.shape}")
    print(f"价值输出形状: {value.shape}")
    print(f"策略概率和: {policy_probs.sum(dim=1)}")
    print(f"价值范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
    
    # 测试单次预测
    single_board = np.random.rand(14, 10, 9).astype(np.float32)
    single_mask = np.ones(8100, dtype=np.float32)
    
    policy, val = net.predict(single_board, single_mask)
    print(f"单次预测 - 策略形状: {policy.shape}, 价值: {val:.3f}")
    
    print("网络测试完成！")


if __name__ == "__main__":
    test_network()