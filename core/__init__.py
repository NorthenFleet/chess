# chess/core/__init__.py

"""
中国象棋游戏核心模块

该模块包含游戏的核心数据结构和基础功能，为其他模块提供基础支持。
"""

from .game_state import GameState, Player, GameEvent, EventType

__all__ = ['GameState', 'Player', 'GameEvent', 'EventType']