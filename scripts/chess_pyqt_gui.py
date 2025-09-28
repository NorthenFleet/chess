#!/usr/bin/env python3
"""
中国象棋PyQt GUI启动脚本
使用PyQt5创建现代化的图形用户界面
"""

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

# 导入游戏引擎和前端组件
from core.game_state import GameState, Player
from board.board import Board, Position
from rule.rule import Rule
from render.ui.main_window import ChessMainWindow

# 定义玩家枚举
class PlayerSide:
    RED = 'red'
    BLACK = 'black'

class ChessGameEngine:
    """简化的游戏引擎适配器"""
    
    def __init__(self):
        self.game_state = GameState()
        self.board = Board()
        self.rule = Rule(self.board)
        self.current_player = PlayerSide.RED
        self.game_over = False
        self.winner = None
        
        # 初始化棋盘
        self.reset_game()
    
    def reset_game(self):
        """重置游戏"""
        self.game_state = GameState()
        self.board = Board()
        self.rule = Rule(self.board)
        self.current_player = PlayerSide.RED
        self.game_over = False
        self.winner = None
        
        # 设置初始棋盘状态
        self.rule.initialize_board()
    
    def get_game_state(self):
        """获取游戏状态"""
        return {
            'board': self.get_board_state(),
            'current_player': 'red' if self.current_player == PlayerSide.RED else 'black',
            'game_over': self.game_over,
            'winner': self.winner
        }
    
    def get_board_state(self):
        """获取棋盘状态"""
        board_state = {}
        
        # 遍历棋盘获取所有棋子
        for row in range(10):
            for col in range(9):
                pos = Position(col, row)  # Position使用(x, y)坐标
                piece = self.board.get_piece_at(pos)
                if piece:
                    piece_info = {
                        'type': self._get_piece_type_name(piece),
                        'color': 'red' if piece.side == 'red' else 'black',
                        'position': (row, col)
                    }
                    board_state[(row, col)] = piece_info
        
        return board_state
    
    def _get_piece_type_name(self, piece):
        """获取棋子类型名称"""
        from piece.piece import PieceType
        
        # 直接使用棋子的type属性进行映射
        type_mapping = {
            PieceType.GENERAL: 'king',
            PieceType.ADVISOR: 'advisor', 
            PieceType.ELEPHANT: 'elephant',
            PieceType.HORSE: 'horse',
            PieceType.CHARIOT: 'rook',  # 车映射为rook
            PieceType.CANNON: 'cannon',
            PieceType.SOLDIER: 'pawn'   # 兵/卒映射为pawn
        }
        
        return type_mapping.get(piece.type, 'unknown')
    
    def make_move(self, from_pos, to_pos):
        """执行移动"""
        try:
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            
            # 转换为Position对象
            from_position = Position(from_col, from_row)
            to_position = Position(to_col, to_row)
            
            # 获取要移动的棋子
            piece = self.board.get_piece_at(from_position)
            if not piece:
                return False
            
            # 检查是否是当前玩家的棋子
            current_side = 'red' if self.current_player == PlayerSide.RED else 'black'
            if piece.side != current_side:
                return False
            
            # 检查移动是否合法
            is_valid, error_msg = self.rule.validate_move(from_position, to_position)
            if not is_valid:
                print(f"移动无效: {error_msg}")
                return False
            
            # 执行移动
            result, captured_piece = self.rule.execute_move(from_position, to_position)
            
            # 检查是否将军或游戏结束
            self._check_game_status()
            
            # 切换玩家
            self.current_player = PlayerSide.BLACK if self.current_player == PlayerSide.RED else PlayerSide.RED
            
            return True
            
        except Exception as e:
            print(f"移动失败: {e}")
            return False
    
    def _check_game_status(self):
        """检查游戏状态"""
        # 简化的游戏结束检查
        # 这里可以添加更复杂的游戏结束逻辑
        pass
    
    def is_game_over(self):
        """检查游戏是否结束"""
        return self.game_over
    
    def get_game_result(self):
        """获取游戏结果"""
        if self.game_over:
            return {
                'winner': 'red' if self.winner == PlayerSide.RED else 'black',
                'reason': '将军'
            }
        return {}

def create_application():
    """创建QApplication实例"""
    app = QApplication(sys.argv)
    app.setApplicationName("中国象棋")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Chess AI Project")
    
    # 设置应用程序图标（如果有的话）
    # app.setWindowIcon(QIcon('path/to/icon.png'))
    
    return app

def main():
    """主函数"""
    try:
        # 创建应用程序
        app = create_application()
        
        # 创建游戏引擎
        game_engine = ChessGameEngine()
        
        # 创建主窗口
        main_window = ChessMainWindow(game_engine)
        main_window.show()
        
        # 显示欢迎消息
        QMessageBox.information(main_window, "欢迎", 
                               "欢迎使用中国象棋游戏！\n\n"
                               "点击棋子选择，再点击目标位置移动。\n"
                               "红方先行，祝您游戏愉快！")
        
        # 运行应用程序
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装PyQt5: pip install PyQt5")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()