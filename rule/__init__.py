# chess/rule/__init__.py

"""
中国象棋规则模块

该模块负责游戏规则的实现，包括走子验证、将军判定和胜负判定。
"""

from .rule import Rule, MoveResult

__all__ = ['Rule', 'MoveResult']