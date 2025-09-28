"""
PyQt前端资源模块
包含棋子图像、样式表等资源文件
"""

from .chess_pieces import get_piece_svg, get_all_pieces, RED_PIECES, BLACK_PIECES

__all__ = [
    'get_piece_svg',
    'get_all_pieces', 
    'RED_PIECES',
    'BLACK_PIECES'
]