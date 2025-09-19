# chess/main.py

import os
import sys
import time
from typing import Tuple, Optional, Dict, List

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.game_state import GameState, Player
from board.board import Board, Position
from piece.piece import Piece, PieceType
from rule.rule import Rule, MoveResult


class ChineseChess:
    """中国象棋游戏主类"""
    
    def __init__(self):
        """初始化游戏"""
        self.board = Board()
        self.rule = Rule(self.board)
        self.game_state = GameState()
        
        # 初始化玩家
        self.game_state.setup_players("红方", "黑方")
        
        # 初始化棋盘
        self.rule.initialize_board()
        
        # 开始游戏（设置当前玩家为红方）
        self.game_state.start_game()
    
    def display_board(self):
        """显示棋盘"""
        # 清屏
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # 打印列坐标
        print("  " + " ".join([str(i) for i in range(9)]))
        
        # 打印棋盘内容
        for y in range(9, -1, -1):
            row = f"{y} "
            for x in range(9):
                pos = Position(x, y)
                piece = self.board.get_piece_at(pos)
                if piece:
                    row += piece.symbol + " "
                else:
                    # 在特殊位置显示标记
                    if (x == 0 or x == 8) and (y == 0 or y == 9):
                        row += "┌ "  # 角落
                    elif (x == 0 or x == 8) and (y == 4 or y == 5):
                        row += "┼ "  # 河界
                    elif x == 4 and (y == 0 or y == 9):
                        row += "╬ "  # 中心点
                    elif x % 2 == 0 and y % 3 == 0:
                        row += "┼ "  # 交叉点
                    else:
                        row += "· "  # 普通点
            print(row)
        
        # 打印当前玩家
        current_player = self.game_state.current_player
        print(f"\n当前回合: {current_player.name} ({current_player.side})")
    
    def parse_position(self, pos_str: str) -> Optional[Position]:
        """解析用户输入的位置"""
        try:
            # 格式应为"x,y"，例如"0,0"
            x, y = map(int, pos_str.strip().split(','))
            if 0 <= x <= 8 and 0 <= y <= 9:
                return Position(x, y)
            else:
                print("位置超出棋盘范围！")
                return None
        except ValueError:
            print("位置格式错误！请使用'x,y'格式，例如'0,0'。")
            return None
    
    def get_user_move(self) -> Tuple[Optional[Position], Optional[Position]]:
        """获取用户的移动"""
        while True:
            # 获取起始位置
            from_str = input("请输入起始位置 (x,y)，或输入'q'退出: ")
            if from_str.lower() == 'q':
                return None, None
            
            from_pos = self.parse_position(from_str)
            if not from_pos:
                continue
            
            # 检查起始位置是否有当前玩家的棋子
            piece = self.board.get_piece_at(from_pos)
            current_player = self.game_state.current_player
            if not piece:
                print("该位置没有棋子！")
                continue
            if piece.side != current_player.side:
                print(f"该位置的棋子不属于当前玩家 {current_player.name}！")
                continue
            
            # 获取目标位置
            to_str = input("请输入目标位置 (x,y)，或输入'c'取消选择: ")
            if to_str.lower() == 'c':
                continue
            
            to_pos = self.parse_position(to_str)
            if not to_pos:
                continue
            
            return from_pos, to_pos
    
    def play(self):
        """开始游戏"""
        print("欢迎来到中国象棋！")
        print("输入位置格式为'x,y'，例如'0,0'表示左下角。")
        print("红方先行。\n")
        
        game_over = False
        while not game_over:
            # 显示棋盘
            self.display_board()
            
            # 获取当前玩家的移动
            from_pos, to_pos = self.get_user_move()
            if not from_pos or not to_pos:  # 用户退出
                print("游戏已退出。")
                return
            
            # 执行移动
            result, captured = self.rule.execute_move(from_pos, to_pos)
            
            # 处理移动结果
            if result == MoveResult.INVALID:
                print("无效的移动！请重试。")
                time.sleep(1)
                continue
            
            # 显示移动结果
            current_player = self.game_state.current_player
            if result == MoveResult.CAPTURE:
                print(f"{current_player.name} 吃掉了 {captured.symbol}")
            elif result == MoveResult.CHECK:
                print(f"{current_player.name} 将军！")
            elif result == MoveResult.CHECKMATE:
                print(f"{current_player.name} 将死！游戏结束。")
                game_over = True
            
            # 检查游戏是否结束
            is_over, winner, reason = self.rule.is_game_over()
            if is_over:
                self.display_board()
                if winner:
                    winner_player = self.game_state.players[winner]
                    print(f"游戏结束！{winner_player.name} 获胜！原因：{reason}")
                else:
                    print(f"游戏结束！和棋！原因：{reason}")
                game_over = True
                break
            
            # 切换玩家
            self.game_state.switch_turn()
        
        print("感谢游玩！")


if __name__ == "__main__":
    game = ChineseChess()
    game.play()