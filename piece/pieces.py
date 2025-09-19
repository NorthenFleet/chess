# chess/piece/pieces.py

import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from typing import List, Set, Optional
from piece.piece import Piece, PieceType
from board.board import Position, Board


class General(Piece):
    """将/帅棋子类"""
    
    def __init__(self, side: str):
        super().__init__(side, PieceType.GENERAL)
    
    def get_valid_moves(self, position: Position, board: Board) -> List[Position]:
        """获取将/帅的合法移动位置"""
        valid_moves = []
        
        # 将/帅只能在九宫格内移动
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上、右、下、左
        
        for dx, dy in directions:
            new_x, new_y = position.x + dx, position.y + dy
            
            # 检查是否在棋盘范围内
            if not (0 <= new_x <= 8 and 0 <= new_y <= 9):
                continue
                
            new_pos = Position(new_x, new_y)
            
            # 检查是否在九宫格内
            if board.is_in_palace(new_pos, self.side):
                # 检查目标位置是否有己方棋子
                piece_at_target = board.get_piece_at(new_pos)
                if not piece_at_target or piece_at_target.side != self.side:
                    valid_moves.append(new_pos)
        
        return valid_moves


class Advisor(Piece):
    """士/仕棋子类"""
    
    def __init__(self, side: str):
        super().__init__(side, PieceType.ADVISOR)
    
    def get_valid_moves(self, position: Position, board: Board) -> List[Position]:
        """获取士/仕的合法移动位置"""
        valid_moves = []
        
        # 士/仕只能在九宫格内斜线移动
        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]  # 右上、右下、左下、左上
        
        for dx, dy in directions:
            new_x, new_y = position.x + dx, position.y + dy
            
            # 检查是否在棋盘范围内
            if not (0 <= new_x <= 8 and 0 <= new_y <= 9):
                continue
                
            new_pos = Position(new_x, new_y)
            
            # 检查是否在九宫格内
            if board.is_in_palace(new_pos, self.side):
                # 检查目标位置是否有己方棋子
                piece_at_target = board.get_piece_at(new_pos)
                if not piece_at_target or piece_at_target.side != self.side:
                    valid_moves.append(new_pos)
        
        return valid_moves


class Elephant(Piece):
    """象/相棋子类"""
    
    def __init__(self, side: str):
        super().__init__(side, PieceType.ELEPHANT)
    
    def get_valid_moves(self, position: Position, board: Board) -> List[Position]:
        """获取象/相的合法移动位置"""
        valid_moves = []
        
        # 象/相走田字，即斜线走两格
        directions = [(2, 2), (2, -2), (-2, -2), (-2, 2)]  # 右上、右下、左下、左上
        
        for dx, dy in directions:
            new_x, new_y = position.x + dx, position.y + dy
            
            # 检查是否在棋盘范围内
            if not (0 <= new_x <= 8 and 0 <= new_y <= 9):
                continue
                
            new_pos = Position(new_x, new_y)
            
            # 检查是否不过河
            if not board.is_across_river(new_pos, self.side):
                
                # 检查象眼位置是否有棋子（塞象眼）
                eye_x, eye_y = position.x + dx//2, position.y + dy//2
                
                # 检查象眼位置是否在棋盘范围内
                if not (0 <= eye_x <= 8 and 0 <= eye_y <= 9):
                    continue
                    
                eye_pos = Position(eye_x, eye_y)
                
                if not board.get_piece_at(eye_pos):
                    # 检查目标位置是否有己方棋子
                    piece_at_target = board.get_piece_at(new_pos)
                    if not piece_at_target or piece_at_target.side != self.side:
                        valid_moves.append(new_pos)
        
        return valid_moves


class Horse(Piece):
    """马/馬棋子类"""
    
    def __init__(self, side: str):
        super().__init__(side, PieceType.HORSE)
    
    def get_valid_moves(self, position: Position, board: Board) -> List[Position]:
        """获取马/馬的合法移动位置"""
        valid_moves = []
        
        # 马走日，先走一步直线，再走一步斜线
        # 八个可能的方向
        directions = [
            (1, 2), (2, 1),    # 右上
            (2, -1), (1, -2),  # 右下
            (-1, -2), (-2, -1),  # 左下
            (-2, 1), (-1, 2)   # 左上
        ]
        
        for dx, dy in directions:
            new_x, new_y = position.x + dx, position.y + dy
            
            # 检查是否在棋盘范围内
            if not (0 <= new_x <= 8 and 0 <= new_y <= 9):
                continue
                
            new_pos = Position(new_x, new_y)
            
            # 检查马腿位置是否有棋子（蹩马腿）
            leg_x, leg_y = position.x, position.y
            if abs(dx) == 1:  # 横向走一步，纵向走两步
                leg_y += dy // 2
            else:  # 纵向走一步，横向走两步
                leg_x += dx // 2
            
            # 检查马腿位置是否在棋盘范围内
            if not (0 <= leg_x <= 8 and 0 <= leg_y <= 9):
                continue
                
            leg_pos = Position(leg_x, leg_y)
             
            if not board.get_piece_at(leg_pos):
                # 检查目标位置是否有己方棋子
                piece_at_target = board.get_piece_at(new_pos)
                if not piece_at_target or piece_at_target.side != self.side:
                    valid_moves.append(new_pos)
        
        return valid_moves


class Chariot(Piece):
    """车/車棋子类"""
    
    def __init__(self, side: str):
        super().__init__(side, PieceType.CHARIOT)
    
    def get_valid_moves(self, position: Position, board: Board) -> List[Position]:
        """获取车/車的合法移动位置"""
        valid_moves = []
        
        # 车可以横向或纵向移动任意距离，直到遇到棋子或棋盘边缘
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上、右、下、左
        
        for dx, dy in directions:
            # 沿着方向移动，直到遇到障碍
            step = 1
            while True:
                new_x, new_y = position.x + dx * step, position.y + dy * step
                
                # 检查是否超出棋盘范围
                if not (0 <= new_x <= 8 and 0 <= new_y <= 9):
                    break
                    
                new_pos = Position(new_x, new_y)
                
                # 检查目标位置是否有棋子
                piece_at_target = board.get_piece_at(new_pos)
                if not piece_at_target:
                    # 空位，可以移动
                    valid_moves.append(new_pos)
                    step += 1
                elif piece_at_target.side != self.side:
                    # 对方棋子，可以吃
                    valid_moves.append(new_pos)
                    break
                else:
                    # 己方棋子，不能移动
                    break
        
        return valid_moves


class Cannon(Piece):
    """炮/砲棋子类"""
    
    def __init__(self, side: str):
        super().__init__(side, PieceType.CANNON)
    
    def get_valid_moves(self, position: Position, board: Board) -> List[Position]:
        """获取炮/砲的合法移动位置"""
        valid_moves = []
        
        # 炮的移动规则：横向或纵向移动任意距离
        # 吃子时需要一个炮架（中间必须有且仅有一个棋子）
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上、右、下、左
        
        for dx, dy in directions:
            # 沿着方向移动，记录炮架
            step = 1
            has_platform = False
            
            while True:
                new_x, new_y = position.x + dx * step, position.y + dy * step
                
                # 检查是否超出棋盘范围
                if not (0 <= new_x <= 8 and 0 <= new_y <= 9):
                    break
                    
                new_pos = Position(new_x, new_y)
                
                # 检查目标位置是否有棋子
                piece_at_target = board.get_piece_at(new_pos)
                
                if not has_platform:
                    # 还没有炮架
                    if not piece_at_target:
                        # 空位，可以移动
                        valid_moves.append(new_pos)
                        step += 1
                    else:
                        # 遇到棋子，成为炮架
                        has_platform = True
                        step += 1
                else:
                    # 已有炮架
                    if piece_at_target:
                        # 遇到第二个棋子
                        if piece_at_target.side != self.side:
                            # 对方棋子，可以吃
                            valid_moves.append(new_pos)
                        break
                    else:
                        # 空位，继续寻找可能的目标
                        step += 1
        
        return valid_moves


class Soldier(Piece):
    """兵/卒棋子类"""
    
    def __init__(self, side: str):
        super().__init__(side, PieceType.SOLDIER)
    
    def get_valid_moves(self, position: Position, board: Board) -> List[Position]:
        """获取兵/卒的合法移动位置"""
        valid_moves = []
        
        # 兵/卒的移动规则：
        # 1. 未过河前只能向前移动
        # 2. 过河后可以向前或向左右移动
        
        # 根据棋子方向确定前进方向
        # 红方兵向上移动（y增加），黑方卒向下移动（y减少）
        forward_dy = 1 if self.side == 'red' else -1
        
        # 向前移动
        forward_y = position.y + forward_dy
        
        # 检查前进位置是否在棋盘范围内
        if 0 <= forward_y <= 9:
            forward_pos = Position(position.x, forward_y)
            piece_at_target = board.get_piece_at(forward_pos)
            if not piece_at_target or piece_at_target.side != self.side:
                valid_moves.append(forward_pos)
        
        # 检查是否过河
        if board.is_across_river(position, self.side):
            # 过河后可以向左右移动
            for dx in [-1, 1]:
                side_x = position.x + dx
                
                # 检查左右位置是否在棋盘范围内
                if 0 <= side_x <= 8:
                    side_pos = Position(side_x, position.y)
                    piece_at_target = board.get_piece_at(side_pos)
                    if not piece_at_target or piece_at_target.side != self.side:
                        valid_moves.append(side_pos)
        
        return valid_moves