# chess/board/__init__.py

"""
中国象棋棋盘模块

该模块负责棋盘的表示和位置管理，提供棋盘初始化、坐标转换等功能。
"""

from .board import Board, Position

__all__ = ['Board', 'Position']