# 中国象棋游戏使用说明

## 🎮 游戏简介
这是一个完整的中国象棋游戏程序，支持完整的象棋规则，包括将军、将死、和棋等判断。

## 🚀 如何运行游戏

### 方法一：推荐方式（从项目根目录运行）
```bash
cd /Users/sunyi/WorkSpace
python -m chess.main
```

### 方法二：使用启动脚本
```bash
cd /Users/sunyi/WorkSpace/chess
python start_chess.py
```

## 🎯 游戏操作说明

### 输入格式
- 位置格式：使用坐标表示，如 `a0`, `b1`, `c2` 等
- 移动格式：输入起始位置和目标位置，如 `a0 a1`
- 退出游戏：输入 `quit` 或 `exit`

### 坐标系统
```
  a b c d e f g h i
0 車 馬 象 士 將 士 象 馬 車  0
1 . . . . . . . . .  1
2 . 炮 . . . . . 炮 .  2
3 兵 . 兵 . 兵 . 兵 . 兵  3
4 . . . . . . . . .  4
5 . . . . . . . . .  5
6 兵 . 兵 . 兵 . 兵 . 兵  6
7 . 炮 . . . . . 炮 .  7
8 . . . . . . . . .  8
9 車 馬 象 士 帥 士 象 馬 車  9
  a b c d e f g h i
```

### 游戏规则
- 红方先行（下方）
- 各种棋子按照标准中国象棋规则移动
- 将军时必须应将
- 将死或和棋时游戏结束

## 🧪 测试功能

### 运行所有测试
```bash
cd /Users/sunyi/WorkSpace
python -m chess.test_summary
```

### 运行单个测试
```bash
# 基础功能测试
python -m chess.test_chess

# 详细功能测试  
python -m chess.test_detailed

# 胜负判断测试
python -m chess.test_endgame

# 边界情况测试
python -m chess.test_edge_cases
```

## 📁 项目结构
```
chess/
├── board/          # 棋盘相关
├── piece/          # 棋子相关
├── rule/           # 规则相关
├── core/           # 游戏状态管理
├── main.py         # 主程序
├── start_chess.py  # 启动脚本
└── test_*.py       # 测试文件
```

## ⚠️ 注意事项
1. 请确保从正确的目录运行程序
2. 如果遇到模块导入错误，请使用推荐的运行方式
3. 游戏支持完整的中国象棋规则，包括特殊规则（马腿、炮台等）

## 🎉 功能特点
- ✅ 完整的中国象棋规则实现
- ✅ 智能的将军、将死、和棋判断
- ✅ 友好的用户界面
- ✅ 完善的错误处理
- ✅ 全面的测试覆盖

祝您游戏愉快！🎮