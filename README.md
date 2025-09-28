# 中国象棋AI项目

这是一个基于深度学习的中国象棋AI项目，使用PyTorch实现，包含完整的游戏引擎、AI训练系统和用户界面。

## 项目结构

```
chess/
├── ai/                 # AI相关模块
│   ├── algorithms/     # 算法实现（MCTS、PPO等）
│   ├── launchers/      # 训练启动器
│   ├── networks/       # 神经网络模型
│   ├── training/       # 训练相关文件
│   └── utils/          # AI工具函数
├── board/              # 棋盘实现
├── config/             # 配置文件
│   ├── config.py       # 网络和MCTS配置
│   ├── train_config.py # 训练配置
│   └── requirements_ai.txt # AI依赖
├── core/               # 核心游戏逻辑
├── docs/               # 文档
│   ├── README.md       # 详细说明文档
│   ├── README_USAGE.md # 使用指南
│   └── AI_README.md    # AI相关文档
├── logs/               # 日志文件
├── piece/              # 棋子实现
├── rule/               # 游戏规则
├── scripts/            # 启动脚本
│   ├── chess_gui.py    # GUI界面
│   ├── human_vs_ai.py  # 人机对战
│   ├── main.py         # 主程序
│   ├── start_chess.py  # 游戏启动器
│   └── start_training.py # 训练启动器
├── test/               # 测试文件
└── training_data/      # 训练数据
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r config/requirements_ai.txt
```

### 2. 启动游戏

```bash
# 启动图形界面
python scripts/chess_gui.py

# 启动人机对战
python scripts/human_vs_ai.py

# 使用游戏启动器
python scripts/start_chess.py
```

### 3. 训练AI

```bash
# 开始训练
python scripts/start_training.py

# 测试模式训练
python scripts/start_training.py --test

# 从检查点恢复训练
python scripts/start_training.py --resume path/to/checkpoint
```

## 主要功能

- **完整的中国象棋游戏引擎**：支持所有标准规则
- **深度学习AI**：基于神经网络和MCTS的智能对手
- **图形用户界面**：直观的游戏界面
- **AI训练系统**：支持自我对弈训练
- **多种游戏模式**：人人对战、人机对战、AI自对弈

## 技术特性

- 使用PyTorch实现深度学习模型
- 蒙特卡洛树搜索(MCTS)算法
- 近端策略优化(PPO)训练算法
- 模块化设计，易于扩展
- 完整的测试覆盖

## 文档

详细文档请查看 `docs/` 目录：
- [详细说明](docs/README.md)
- [使用指南](docs/README_USAGE.md)
- [AI文档](docs/AI_README.md)

## 许可证

本项目采用开源许可证，具体信息请查看相关文档。