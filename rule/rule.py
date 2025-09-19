# chess/rule/rule.py

import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from board.board import Board, Position
from piece.piece import Piece, PieceType


class MoveResult(Enum):
    """移动结果枚举"""
    VALID = "valid"              # 有效移动
    INVALID = "invalid"          # 无效移动
    CAPTURE = "capture"          # 吃子
    CHECK = "check"              # 将军
    CHECKMATE = "checkmate"      # 将死
    STALEMATE = "stalemate"      # 和棋（无子可动）


class Rule:
    """中国象棋规则类"""
    
    def __init__(self, board: Board):
        """初始化规则对象"""
        self.board = board
    
    def validate_move(self, from_pos: Position, to_pos: Position) -> Tuple[bool, str]:
        """验证移动是否合法"""
        # 检查起始位置是否有棋子
        piece = self.board.get_piece_at(from_pos)
        if not piece:
            return False, "起始位置没有棋子"
        
        # 检查目标位置是否是合法移动
        if not piece.can_move_to(from_pos, to_pos, self.board):
            return False, "不符合棋子走法规则"
        
        # 检查移动后是否会导致己方被将军
        if self.will_be_checked_after_move(from_pos, to_pos, piece.side):
            return False, "移动后会被将军"
        
        return True, "移动合法"
    
    def execute_move(self, from_pos: Position, to_pos: Position) -> Tuple[MoveResult, Optional[Piece]]:
        """执行移动并返回结果"""
        # 验证移动
        is_valid, message = self.validate_move(from_pos, to_pos)
        if not is_valid:
            return MoveResult.INVALID, None
        
        # 获取起始位置的棋子
        piece = self.board.get_piece_at(from_pos)
        
        # 执行移动，可能会吃子
        captured_piece = self.board.move_piece(from_pos, to_pos)
        
        # 检查是否将军
        opponent_side = "black" if piece.side == "red" else "red"
        is_check = self.is_checked(opponent_side)
        
        # 检查是否将死
        is_checkmate = False
        if is_check:
            is_checkmate = self.is_checkmate(opponent_side)
        
        # 确定移动结果
        if is_checkmate:
            result = MoveResult.CHECKMATE
        elif is_check:
            result = MoveResult.CHECK
        elif captured_piece:
            result = MoveResult.CAPTURE
        else:
            result = MoveResult.VALID
        
        return result, captured_piece
    
    def is_checked(self, side: str) -> bool:
        """检查指定方是否被将军"""
        # 找到将/帅的位置
        general_pos = None
        for pos, piece in self.board.pieces.items():
            if piece.type == PieceType.GENERAL and piece.side == side:
                general_pos = pos
                break
        
        if not general_pos:
            return False  # 没有找到将/帅，可能是测试场景
        
        # 检查对方所有棋子是否可以吃掉将/帅
        opponent_side = "black" if side == "red" else "red"
        for pos, piece in self.board.pieces.items():
            if piece.side == opponent_side:
                if piece.can_move_to(pos, general_pos, self.board):
                    return True
        
        return False
    
    def will_be_checked_after_move(self, from_pos: Position, to_pos: Position, side: str) -> bool:
        """检查移动后是否会导致己方被将军"""
        # 保存当前棋盘状态
        original_to_piece = self.board.get_piece_at(to_pos)
        piece = self.board.get_piece_at(from_pos)
        
        # 临时执行移动
        self.board.move_piece(from_pos, to_pos)
        
        # 检查是否被将军
        checked = self.is_checked(side)
        
        # 恢复棋盘状态
        self.board.move_piece(to_pos, from_pos)  # 移回原位置
        if original_to_piece:  # 恢复可能被吃掉的棋子
            self.board.place_piece(original_to_piece, to_pos)
        
        return checked
    
    def get_all_valid_moves(self, side: str) -> Dict[Position, List[Position]]:
        """获取指定方所有棋子的所有合法移动"""
        valid_moves = {}
        
        # 创建棋盘状态的副本，防止在迭代过程中修改字典
        pieces_copy = dict(self.board.pieces)
        
        # 遍历所有己方棋子
        for pos, piece in pieces_copy.items():
            if piece.side == side:
                # 获取该棋子的所有可能移动
                piece_valid_moves = []
                for move in piece.get_valid_moves(pos, self.board):
                    # 检查移动后是否会导致己方被将军
                    if not self.will_be_checked_after_move(pos, move, side):
                        piece_valid_moves.append(move)
                
                if piece_valid_moves:
                    valid_moves[pos] = piece_valid_moves
        
        return valid_moves
    
    def is_checkmate(self, side: str) -> bool:
        """检查指定方是否被将死（无子可动）"""
        # 获取所有合法移动
        valid_moves = self.get_all_valid_moves(side)
        
        # 如果没有合法移动，则被将死
        return len(valid_moves) == 0
    
    def is_stalemate(self, side: str) -> bool:
        """检查是否和棋（无子可动但未被将军）"""
        # 检查是否被将军
        if self.is_checked(side):
            return False
        
        # 获取所有合法移动
        valid_moves = self.get_all_valid_moves(side)
        
        # 如果没有合法移动且未被将军，则和棋
        return len(valid_moves) == 0
    
    def is_game_over(self) -> Tuple[bool, Optional[str], str]:
        """检查游戏是否结束"""
        # 检查红方是否被将死
        if self.is_checked("red") and self.is_checkmate("red"):
            return True, "black", "将死"
        
        # 检查黑方是否被将死
        if self.is_checked("black") and self.is_checkmate("black"):
            return True, "red", "将死"
        
        # 检查和棋（双方都无子可动）
        if self.is_stalemate("red") or self.is_stalemate("black"):
            return True, None, "和棋"
        
        return False, None, ""
    
    def initialize_board(self):
        """初始化标准开局棋盘"""
        from piece.pieces import General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier
        
        # 清空棋盘
        self.board.clear()
        
        # 放置红方棋子（下方）
        # 车
        self.board.place_piece(Chariot("red"), Position(0, 0))
        self.board.place_piece(Chariot("red"), Position(8, 0))
        # 马
        self.board.place_piece(Horse("red"), Position(1, 0))
        self.board.place_piece(Horse("red"), Position(7, 0))
        # 相
        self.board.place_piece(Elephant("red"), Position(2, 0))
        self.board.place_piece(Elephant("red"), Position(6, 0))
        # 仕
        self.board.place_piece(Advisor("red"), Position(3, 0))
        self.board.place_piece(Advisor("red"), Position(5, 0))
        # 帅
        self.board.place_piece(General("red"), Position(4, 0))
        # 炮
        self.board.place_piece(Cannon("red"), Position(1, 2))
        self.board.place_piece(Cannon("red"), Position(7, 2))
        # 兵
        for x in range(0, 9, 2):
            self.board.place_piece(Soldier("red"), Position(x, 3))
        
        # 放置黑方棋子（上方）
        # 车
        self.board.place_piece(Chariot("black"), Position(0, 9))
        self.board.place_piece(Chariot("black"), Position(8, 9))
        # 马
        self.board.place_piece(Horse("black"), Position(1, 9))
        self.board.place_piece(Horse("black"), Position(7, 9))
        # 象
        self.board.place_piece(Elephant("black"), Position(2, 9))
        self.board.place_piece(Elephant("black"), Position(6, 9))
        # 士
        self.board.place_piece(Advisor("black"), Position(3, 9))
        self.board.place_piece(Advisor("black"), Position(5, 9))
        # 将
        self.board.place_piece(General("black"), Position(4, 9))
        # 炮
        self.board.place_piece(Cannon("black"), Position(1, 7))
        self.board.place_piece(Cannon("black"), Position(7, 7))
        # 卒
        for x in range(0, 9, 2):
            self.board.place_piece(Soldier("black"), Position(x, 6))