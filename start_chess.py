#!/usr/bin/env python3
"""
中国象棋游戏启动脚本
解决模块导入问题的启动器
"""

import sys
import os

# 添加上级目录到Python路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """启动中国象棋游戏"""
    try:
        print("=" * 60)
        print("🎮 中国象棋游戏")
        print("=" * 60)
        print("正在启动游戏...")
        
        # 导入并启动游戏
        from chess.main import ChineseChess
        
        print("✓ 游戏模块加载成功")
        
        # 创建并启动游戏
        game = ChineseChess()
        print("✓ 游戏实例创建成功")
        print("\n开始游戏！")
        print("=" * 60)
        
        # 启动游戏主循环
        game.play()
        
    except KeyboardInterrupt:
        print("\n\n游戏被用户中断")
        print("感谢游玩！👋")
    except ImportError as e:
        print(f"❌ 模块导入错误: {e}")
        print("请确保所有必要的文件都在正确位置")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 游戏运行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()