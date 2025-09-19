# chess/board/board.py

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class Position:
    """棋盘位置类"""
    x: int  # 横坐标 (0-8)
    y: int  # 纵坐标 (0-9)
    
    def __post_init__(self):
        # 验证坐标范围
        if not (0 <= self.x <= 8):
            raise ValueError(f"横坐标 x 必须在 0-8 范围内，当前值: {self.x}")
        if not (0 <= self.y <= 9):
            raise ValueError(f"纵坐标 y 必须在 0-9 范围内，当前值: {self.y}")
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self):
        # 转换为中文坐标表示法
        file_names = ['九', '八', '七', '六', '五', '四', '三', '二', '一']
        rank_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        return f"{file_names[self.x]}{rank_names[self.y]}"
    
    def to_notation(self) -> str:
        """转换为代数记号 (例如: 'a1', 'e5')"""
        file_names = 'abcdefghi'
        rank_names = '0123456789'
        return f"{file_names[self.x]}{rank_names[self.y]}"
    
    @classmethod
    def from_notation(cls, notation: str) -> 'Position':
        """从代数记号创建位置对象"""
        if len(notation) != 2 or notation[0] not in 'abcdefghi' or notation[1] not in '0123456789':
            raise ValueError(f"无效的位置记号: {notation}")
        
        file_names = 'abcdefghi'
        rank_names = '0123456789'
        x = file_names.index(notation[0])
        y = rank_names.index(notation[1])
        return cls(x, y)


class Board:
    """中国象棋棋盘类"""
    
    # 棋盘尺寸常量
    WIDTH = 9  # 横线数量
    HEIGHT = 10  # 纵线数量
    
    # 特殊区域
    PALACE_RED = {Position(3, 0), Position(4, 0), Position(5, 0),
                  Position(3, 1), Position(4, 1), Position(5, 1),
                  Position(3, 2), Position(4, 2), Position(5, 2)}
    
    PALACE_BLACK = {Position(3, 7), Position(4, 7), Position(5, 7),
                    Position(3, 8), Position(4, 8), Position(5, 8),
                    Position(3, 9), Position(4, 9), Position(5, 9)}
    
    RIVER_LINE = 4  # 河界位置（y坐标）
    
    def __init__(self):
        """初始化空棋盘"""
        # 棋盘上的棋子，键为位置，值为棋子对象
        self.pieces = {}
        # 棋子位置索引，键为棋子id，值为位置
        self.piece_positions = {}
    
    def is_valid_position(self, pos: Position) -> bool:
        """检查位置是否在棋盘范围内"""
        return 0 <= pos.x < self.WIDTH and 0 <= pos.y < self.HEIGHT
    
    def is_in_palace(self, pos: Position, side: str) -> bool:
        """检查位置是否在九宫格内"""
        if side.lower() == 'red':
            return pos in self.PALACE_RED
        elif side.lower() == 'black':
            return pos in self.PALACE_BLACK
        return False
    
    def is_across_river(self, pos: Position, side: str) -> bool:
        """检查位置是否过河"""
        if side.lower() == 'red':
            return pos.y > self.RIVER_LINE
        elif side.lower() == 'black':
            return pos.y < self.RIVER_LINE
        return False
    
    def get_piece_at(self, pos: Position):
        """获取指定位置的棋子"""
        return self.pieces.get(pos)
    
    def place_piece(self, piece, pos: Position) -> None:
        """在指定位置放置棋子"""
        if not self.is_valid_position(pos):
            raise ValueError(f"无效的棋盘位置: {pos}")
        
        # 如果该位置已有棋子，先移除
        if pos in self.pieces:
            old_piece = self.pieces[pos]
            del self.piece_positions[old_piece.id]
        
        # 放置新棋子
        self.pieces[pos] = piece
        self.piece_positions[piece.id] = pos
    
    def remove_piece(self, pos: Position) -> Optional[object]:
        """移除指定位置的棋子"""
        if pos in self.pieces:
            piece = self.pieces[pos]
            del self.pieces[pos]
            del self.piece_positions[piece.id]
            return piece
        return None
    
    def move_piece(self, from_pos: Position, to_pos: Position) -> Optional[object]:
        """移动棋子，返回被吃的棋子（如果有）"""
        if not self.is_valid_position(from_pos) or not self.is_valid_position(to_pos):
            raise ValueError(f"无效的棋盘位置: {from_pos} -> {to_pos}")
        
        piece = self.get_piece_at(from_pos)
        if not piece:
            raise ValueError(f"起始位置没有棋子: {from_pos}")
        
        # 检查目标位置是否有棋子（可能被吃）
        captured = self.remove_piece(to_pos)
        
        # 移动棋子
        del self.pieces[from_pos]
        self.pieces[to_pos] = piece
        self.piece_positions[piece.id] = to_pos
        
        return captured
    
    def get_position(self, piece_id: str) -> Optional[Position]:
        """获取指定棋子的位置"""
        return self.piece_positions.get(piece_id)
    
    def clear(self) -> None:
        """清空棋盘"""
        self.pieces.clear()
        self.piece_positions.clear()
    
    def get_pieces_by_side(self, side: str) -> Dict[Position, object]:
        """获取指定方的所有棋子"""
        return {pos: piece for pos, piece in self.pieces.items() if piece.side.lower() == side.lower()}
    
    def to_fen(self) -> str:
        """转换为FEN记号（Forsyth-Edwards Notation）"""
        # 简化版FEN，仅表示棋子位置
        fen_rows = []
        for y in range(self.HEIGHT - 1, -1, -1):  # 从上到下（黑方到红方）
            row = ""
            empty_count = 0
            
            for x in range(self.WIDTH):
                pos = Position(x, y)
                piece = self.get_piece_at(pos)
                
                if piece:
                    if empty_count > 0:
                        row += str(empty_count)
                        empty_count = 0
                    row += piece.fen_symbol
                else:
                    empty_count += 1
            
            if empty_count > 0:
                row += str(empty_count)
            
            fen_rows.append(row)
        
        return "/".join(fen_rows)
    
    def __str__(self) -> str:
        """返回棋盘的字符串表示"""
        result = []
        # 添加列标签
        result.append("  a b c d e f g h i")
        result.append("  ---------------")
        
        for y in range(self.HEIGHT - 1, -1, -1):
            row = f"{y}|"
            for x in range(self.WIDTH):
                pos = Position(x, y)
                piece = self.get_piece_at(pos)
                if piece:
                    row += piece.symbol + " "
                else:
                    # 在特殊位置添加标记
                    if (x == 1 or x == 7) and y == 2:  # 炮位
                        row += "+ "
                    elif (x == 1 or x == 7) and y == 7:  # 炮位
                        row += "+ "
                    elif x % 2 == 0 and y == 3:  # 兵位
                        row += "+ "
                    elif x % 2 == 0 and y == 6:  # 卒位
                        row += "+ "
                    elif (pos in self.PALACE_RED or pos in self.PALACE_BLACK) and (x == 4 and (y == 0 or y == 9)):  # 将/帅位
                        row += "# "
                    elif (pos in self.PALACE_RED or pos in self.PALACE_BLACK) and ((x == 3 or x == 5) and (y == 0 or y == 2 or y == 7 or y == 9)):  # 士位
                        row += "* "
                    elif ((x == 2 or x == 6) and (y == 0 or y == 9)):  # 象/相位
                        row += "@ "
                    elif ((x == 1 or x == 7) and (y == 0 or y == 9)):  # 马位
                        row += "^ "
                    elif ((x == 0 or x == 8) and (y == 0 or y == 9)):  # 车位
                        row += "$ "
                    else:
                        row += ". "
            
            result.append(row)
        
        return "\n".join(result)