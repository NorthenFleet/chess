"""
PyQt组件模块
包含棋盘、棋子等UI组件
"""

from .chess_board import ChessBoardWidget
from .chess_piece import ChessPieceWidget

__all__ = [
    'ChessBoardWidget',
    'ChessPieceWidget'
]