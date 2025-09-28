"""
PyQt前端渲染模块
提供中国象棋游戏的图形用户界面
"""

__version__ = "1.0.0"
__author__ = "Chess AI Project"

# 导入主要组件
from .ui.main_window import ChessMainWindow
from .components.chess_board import ChessBoardWidget
from .dss_interface import DSSInterface

__all__ = [
    'ChessMainWindow',
    'ChessBoardWidget', 
    'DSSInterface'
]