# chess/piece/piece.py

import os
import sys
import uuid

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, TYPE_CHECKING

from board.board import Position

if TYPE_CHECKING:
    from board.board import Board


class PieceType(Enum):
    """棋子类型枚举"""
    GENERAL = "general"    # 将/帅
    ADVISOR = "advisor"    # 士/仕
    ELEPHANT = "elephant"  # 象/相
    HORSE = "horse"        # 马/馬
    CHARIOT = "chariot"    # 车/車
    CANNON = "cannon"      # 炮/砲
    SOLDIER = "soldier"    # 兵/卒


class Piece:
    """棋子基类"""
    
    def __init__(self, side: str, piece_type: PieceType):
        """
        初始化棋子
        
        Args:
            side: 棋子所属方 ('red' 或 'black')
            piece_type: 棋子类型
        """
        self.id = str(uuid.uuid4())  # 唯一标识符
        self.side = side.lower()  # 'red' 或 'black'
        self.type = piece_type
        self.captured = False  # 是否被吃
        
        # 设置棋子符号和FEN符号
        self._set_symbols()
    
    def _set_symbols(self):
        """设置棋子的显示符号和FEN符号"""
        # 中文名称映射
        chinese_names = {
            PieceType.GENERAL: {"red": "帅", "black": "将"},
            PieceType.ADVISOR: {"red": "仕", "black": "士"},
            PieceType.ELEPHANT: {"red": "相", "black": "象"},
            PieceType.HORSE: {"red": "马", "black": "马"},
            PieceType.CHARIOT: {"red": "车", "black": "车"},
            PieceType.CANNON: {"red": "炮", "black": "炮"},
            PieceType.SOLDIER: {"red": "兵", "black": "卒"}
        }
        
        # FEN符号映射
        fen_symbols = {
            PieceType.GENERAL: {"red": "K", "black": "k"},
            PieceType.ADVISOR: {"red": "A", "black": "a"},
            PieceType.ELEPHANT: {"red": "E", "black": "e"},
            PieceType.HORSE: {"red": "H", "black": "h"},
            PieceType.CHARIOT: {"red": "R", "black": "r"},
            PieceType.CANNON: {"red": "C", "black": "c"},
            PieceType.SOLDIER: {"red": "P", "black": "p"}
        }
        
        self.name = chinese_names[self.type][self.side]
        self.symbol = self.name
        self.fen_symbol = fen_symbols[self.type][self.side]
    
    def get_valid_moves(self, position: Position, board) -> List[Position]:
        """获取当前位置下的所有合法移动位置"""
        # 基类中提供默认实现，子类应重写此方法
        return []
    
    def can_move_to(self, from_pos: Position, to_pos: Position, board) -> bool:
        """检查是否可以从当前位置移动到目标位置"""
        valid_moves = self.get_valid_moves(from_pos, board)
        return to_pos in valid_moves
    
    def __str__(self) -> str:
        return f"{self.side.capitalize()} {self.type.value}: {self.name}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.side} {self.id[:8]}>"