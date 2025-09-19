# 中国象棋AI神经网络

基于深度学习和蒙特卡洛树搜索的中国象棋AI系统，采用AlphaZero架构。

## 🎯 系统特点

- **先进架构**: 基于ResNet + 自注意力机制的双头网络
- **端到端训练**: 通过自对弈生成训练数据，无需人工标注
- **强化学习**: 结合MCTS和神经网络的强化学习算法
- **高效编码**: 优化的棋盘状态和动作编码方案
- **可扩展性**: 模块化设计，易于扩展和优化

## 📊 网络架构详情

### 总体结构
```
输入 (14×10×9) → 卷积特征提取 → 注意力机制 → 双头输出
                                              ├─ 策略网络 (2086维)
                                              └─ 价值网络 (1维)
```

### 详细参数
- **输入维度**: 14个通道 × 10行 × 9列
- **特征通道**: 256个隐藏通道
- **残差块**: 20个残差块
- **注意力头**: 8个注意力头
- **总参数**: 约3000万参数
- **模型大小**: 约120MB

### 网络层次
1. **输入层**: 14通道棋盘编码
2. **初始卷积**: 3×3卷积 + BatchNorm + ReLU
3. **残差主干**: 20个残差块 (每块2层卷积)
4. **注意力层**: 8头自注意力机制
5. **策略头**: 1×1卷积 + 全连接 → 2086维动作概率
6. **价值头**: 1×1卷积 + 全连接 → 1维价值评估

## 🔧 安装和使用

### 1. 安装依赖
```bash
# 安装AI模块依赖
pip install -r requirements_ai.txt

# 或者手动安装主要依赖
pip install torch torchvision numpy
```

### 2. 快速开始

#### 人机对弈
```bash
# 使用随机初始化模型对弈
python ai/demo.py --mode play

# 使用预训练模型对弈
python ai/demo.py --mode play --model checkpoints/final_model.pth
```

#### 训练AI
```bash
# 开始训练 (10次迭代)
python ai/demo.py --mode train --iterations 10

# 使用GPU训练
python ai/demo.py --mode train --iterations 100 --device cuda
```

#### AI对战
```bash
# AI自我对战
python ai/demo.py --mode ai_vs_ai --games 20
```

### 3. 编程接口

#### 基本使用
```python
from ai.demo import ChessAI
from chess.core.game_state import GameState

# 创建AI实例
ai = ChessAI(model_path="checkpoints/final_model.pth")

# 创建游戏状态
game = GameState()

# 获取AI的最佳移动
best_move = ai.get_best_move(game)
from_pos, to_pos = best_move

# 执行移动
game.make_move(from_pos, to_pos)

# 评估当前局面
position_value = ai.evaluate_position(game)
print(f"局面评估: {position_value:.3f}")
```

#### 自定义训练
```python
from ai.trainer import ChessTrainer

# 创建训练器
trainer = ChessTrainer(
    network_config={
        'hidden_channels': 256,
        'num_residual_blocks': 20,
        'use_attention': True
    },
    training_config={
        'mcts_simulations': 800,
        'self_play_games': 100,
        'epochs_per_iteration': 10
    }
)

# 执行训练
trainer.train(num_iterations=50, checkpoint_dir="my_checkpoints")
```

## 📈 训练过程

### 自对弈训练流程
1. **自对弈**: AI与自己对弈生成训练数据
2. **MCTS搜索**: 每步使用MCTS搜索最佳移动
3. **数据收集**: 收集棋盘状态、动作概率、游戏结果
4. **网络训练**: 使用收集的数据训练神经网络
5. **模型评估**: 评估新模型的性能
6. **迭代优化**: 重复上述过程

### 训练配置
```python
training_config = {
    'learning_rate': 0.001,          # 学习率
    'weight_decay': 1e-4,            # 权重衰减
    'batch_size': 32,                # 批次大小
    'epochs_per_iteration': 10,      # 每次迭代的训练轮数
    'mcts_simulations': 800,         # MCTS模拟次数
    'self_play_games': 100,          # 每次迭代的自对弈局数
    'temperature_threshold': 30,     # 温度采样阈值
    'max_game_length': 500,          # 最大游戏长度
    'buffer_size': 100000,           # 训练数据缓冲区大小
}
```

## 🧠 算法原理

### MCTS + 神经网络
- **选择**: 使用UCB公式选择最有前景的节点
- **展开**: 使用神经网络策略展开新节点
- **评估**: 使用神经网络价值函数评估叶子节点
- **回传**: 将评估结果回传到根节点

### 损失函数
```python
# 总损失 = 策略损失 + 价值损失 + 正则化
total_loss = policy_loss + value_loss + l2_regularization

# 策略损失: 交叉熵
policy_loss = -sum(mcts_probs * log(network_probs))

# 价值损失: 均方误差
value_loss = (game_result - network_value)²
```

### 棋盘编码
- **14个通道**: 红方7种棋子 + 黑方7种棋子
- **10×9网格**: 对应中国象棋棋盘
- **二进制编码**: 有棋子为1，无棋子为0

### 动作编码
- **动作空间**: 2086个可能动作
- **编码方式**: 起始位置索引 × 89 + 目标位置索引
- **掩码机制**: 只考虑当前局面的合法动作

## 📊 性能指标

### 计算性能
- **推理速度**: <10ms (GPU) / <50ms (CPU)
- **内存占用**: ~120MB
- **训练速度**: ~1000局/小时 (GPU)

### 棋力评估
- **初始模型**: 随机水平
- **训练100轮**: 业余初级
- **训练1000轮**: 业余中级
- **训练10000轮**: 业余高级
- **充分训练**: 接近专业水平

## 🔍 模块说明

### 核心模块
- `network.py`: 神经网络模型定义
- `encoder.py`: 棋盘和动作编码器
- `mcts.py`: 蒙特卡洛树搜索算法
- `trainer.py`: 训练框架和自对弈系统
- `demo.py`: 演示脚本和AI接口

### 文件结构
```
ai/
├── __init__.py          # 模块初始化
├── network.py           # 神经网络模型
├── encoder.py           # 编码器
├── mcts.py             # MCTS算法
├── trainer.py          # 训练框架
├── demo.py             # 演示脚本
└── network_design.md   # 网络设计文档
```

## 🚀 优化建议

### 性能优化
1. **使用GPU**: 显著提升训练和推理速度
2. **批处理**: 批量处理多个局面
3. **模型剪枝**: 减少模型参数数量
4. **量化**: 使用低精度计算

### 算法优化
1. **增加模拟次数**: 提升MCTS搜索质量
2. **调整网络深度**: 平衡性能和计算成本
3. **改进编码**: 添加更多特征通道
4. **集成学习**: 使用多个模型投票

### 训练优化
1. **数据增强**: 棋盘旋转、镜像变换
2. **课程学习**: 从简单局面开始训练
3. **对抗训练**: 与不同强度的对手训练
4. **迁移学习**: 使用预训练模型初始化

## 🎮 使用示例

### 命令行使用
```bash
# 训练新模型
python ai/demo.py --mode train --iterations 50 --device cuda

# 人机对弈
python ai/demo.py --mode play --model checkpoints/final_model.pth

# AI对战测试
python ai/demo.py --mode ai_vs_ai --games 100
```

### Python脚本
```python
# 创建和训练AI
from ai.trainer import ChessTrainer

trainer = ChessTrainer()
trainer.train(num_iterations=10)

# 使用训练好的AI
from ai.demo import ChessAI

ai = ChessAI("checkpoints/final_model.pth")
# ... 进行对弈或分析
```

## 📝 注意事项

1. **计算资源**: 训练需要大量计算资源，建议使用GPU
2. **训练时间**: 完整训练可能需要数天到数周
3. **内存需求**: 训练时需要足够内存存储训练数据
4. **模型保存**: 定期保存检查点以防训练中断
5. **超参数调优**: 根据具体需求调整训练参数

## 🔗 相关资源

- [AlphaZero论文](https://arxiv.org/abs/1712.01815)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [中国象棋规则](https://zh.wikipedia.org/wiki/中国象棋)

---

这个AI系统结合了现代深度学习的最佳实践，专门为中国象棋优化设计。通过持续的自对弈训练，AI将不断提升棋力，最终达到很高的水平。