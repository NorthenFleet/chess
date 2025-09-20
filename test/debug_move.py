# chess/test/debug_move.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board.board import Board, Position
from piece.pieces import Soldier
from rule.rule import Rule

# 创建棋盘和规则
board = Board()
rule = Rule(board)

# 初始化棋盘
rule.initialize_board()

# 检查红兵的位置和移动
red_soldier_pos = Position(0, 3)
piece = board.get_piece_at(red_soldier_pos)

print(f"位置 {red_soldier_pos} 的棋子: {piece}")
print(f"棋子类型: {piece.type if piece else 'None'}")
print(f"棋子方: {piece.side if piece else 'None'}")

if piece:
    valid_moves = piece.get_valid_moves(red_soldier_pos, board)
    print(f"合法移动: {valid_moves}")
    
    # 测试向前移动
    target_pos = Position(0, 4)
    can_move = piece.can_move_to(red_soldier_pos, target_pos, board)
    print(f"能否移动到 {target_pos}: {can_move}")
    
    # 检查移动验证
    is_valid, message = rule.validate_move(red_soldier_pos, target_pos)
    print(f"移动验证结果: {is_valid}, 消息: {message}")

# 显示棋盘状态
print("\n棋盘状态:")
for y in range(9, -1, -1):
    row = f"{y} "
    for x in range(9):
        pos = Position(x, y)
        piece = board.get_piece_at(pos)
        if piece:
            row += piece.symbol + " "
        else:
            row += "· "
    print(row)