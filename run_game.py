#!/usr/bin/env python3
"""
中国象棋游戏启动脚本
用于测试游戏是否能正常启动和运行
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from chess.main import ChineseChess
    
    print("=" * 50)
    print("中国象棋游戏启动测试")
    print("=" * 50)
    
    # 创建游戏实例
    game = ChineseChess()
    print("✓ 游戏实例创建成功")
    
    # 测试棋盘显示
    print("\n当前棋盘状态:")
    game.display_board()
    
    print("\n✓ 游戏启动成功！")
    print("注意: 这是启动测试，实际游戏需要运行 'python -m chess.main'")
    print("=" * 50)
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 运行错误: {e}")
    sys.exit(1)