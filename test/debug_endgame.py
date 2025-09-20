# chess/test/debug_endgame.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board.board import Board, Position
from piece.pieces import General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier
from rule.rule import Rule, MoveResult


def debug_checkmate():
    """调试将死检测"""
    print("=== 调试将死检测 ===")
    
    board = Board()
    rule = Rule(board)
    
    board.clear()
    
    # 设置一个简单的将死局面
    board.place_piece(General("black"), Position(4, 9))
    board.place_piece(General("red"), Position(4, 7))
    board.place_piece(Chariot("red"), Position(3, 9))
    board.place_piece(Chariot("red"), Position(5, 9))
    board.place_piece(Chariot("red"), Position(4, 8))
    
    print("棋盘状态:")
    print(board)
    
    # 检查黑方是否被将军
    is_checked = rule.is_checked("black")
    print(f"黑方被将军: {is_checked}")
    
    # 获取黑方所有合法移动
    valid_moves = rule.get_all_valid_moves("black")
    print(f"黑方合法移动: {valid_moves}")
    
    # 检查是否将死
    is_checkmate = rule.is_checkmate("black")
    print(f"黑方被将死: {is_checkmate}")
    
    # 检查游戏是否结束
    is_over, winner, reason = rule.is_game_over()
    print(f"游戏结束: {is_over}, 获胜方: {winner}, 原因: {reason}")


def debug_move_validation():
    """调试移动验证"""
    print("\n=== 调试移动验证 ===")
    
    board = Board()
    rule = Rule(board)
    
    board.clear()
    board.place_piece(General("red"), Position(4, 0))
    board.place_piece(General("black"), Position(4, 9))
    board.place_piece(Chariot("black"), Position(4, 1))
    
    print("棋盘状态:")
    print(board)
    
    # 尝试移动红将到会被将军的位置
    print("尝试移动红将从 (4,0) 到 (3,0)")
    
    # 检查移动前状态
    print(f"移动前红方被将军: {rule.is_checked('red')}")
    
    # 检查移动后是否会被将军
    will_be_checked = rule.will_be_checked_after_move(Position(4, 0), Position(3, 0), "red")
    print(f"移动后会被将军: {will_be_checked}")
    
    # 执行移动
    result, captured = rule.execute_move(Position(4, 0), Position(3, 0))
    print(f"移动结果: {result}")
    
    print("移动后棋盘状态:")
    print(board)


def debug_stalemate():
    """调试和棋检测"""
    print("\n=== 调试和棋检测 ===")
    
    board = Board()
    rule = Rule(board)
    
    board.clear()
    board.place_piece(General("red"), Position(4, 0))
    board.place_piece(General("black"), Position(4, 2))
    
    print("棋盘状态:")
    print(board)
    
    # 检查双方状态
    print(f"红方被将军: {rule.is_checked('red')}")
    print(f"黑方被将军: {rule.is_checked('black')}")
    
    # 获取合法移动
    red_moves = rule.get_all_valid_moves("red")
    black_moves = rule.get_all_valid_moves("black")
    print(f"红方合法移动: {red_moves}")
    print(f"黑方合法移动: {black_moves}")
    
    # 检查和棋
    red_stalemate = rule.is_stalemate("red")
    black_stalemate = rule.is_stalemate("black")
    print(f"红方和棋: {red_stalemate}")
    print(f"黑方和棋: {black_stalemate}")


if __name__ == "__main__":
    debug_checkmate()
    debug_move_validation()
    debug_stalemate()