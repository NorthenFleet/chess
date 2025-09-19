"""
棋盘状态编码和动作编码模块

负责将棋盘状态转换为神经网络输入，以及动作的编码解码
"""

import numpy as np
import sys
import os
from typing import Tuple, List, Optional

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from board.board import Board, Position
from piece.piece import PieceType


class BoardEncoder:
    """棋盘状态编码器"""
    
    def __init__(self):
        """初始化编码器"""
        # 棋子类型到通道索引的映射
        self.piece_to_channel = {
            # 红方棋子 (通道 0-6)
            ("red", PieceType.GENERAL): 0,      # 帅
            ("red", PieceType.ADVISOR): 1,      # 士  
            ("red", PieceType.ELEPHANT): 2,     # 象
            ("red", PieceType.CHARIOT): 3,      # 车
            ("red", PieceType.HORSE): 4,        # 马
            ("red", PieceType.CANNON): 5,       # 炮
            ("red", PieceType.SOLDIER): 6,      # 兵
            
            # 黑方棋子 (通道 7-13)
            ("black", PieceType.GENERAL): 7,      # 将
            ("black", PieceType.ADVISOR): 8,      # 士
            ("black", PieceType.ELEPHANT): 9,     # 象  
            ("black", PieceType.CHARIOT): 10,     # 车
            ("black", PieceType.HORSE): 11,       # 马
            ("black", PieceType.CANNON): 12,      # 炮
            ("black", PieceType.SOLDIER): 13,     # 兵
        }
        
        # 输入张量形状: (14, 10, 9)
        self.input_shape = (14, 10, 9)

    def encode_board(self, board: Board, current_player: str) -> np.ndarray:
        """
        将棋盘状态编码为神经网络输入
        
        Args:
            board: 棋盘对象
            current_player: 当前玩家颜色 ('red' 或 'black')
            
        Returns:
            编码后的张量，形状为 (14, 10, 9)
        """
        # 初始化输入张量
        encoded = np.zeros(self.input_shape, dtype=np.float32)
        
        # 遍历棋盘上的所有位置
        for row in range(10):  # 10行
            for col in range(9):  # 9列
                pos = Position(col, row)  # Position(x, y) 其中 x 是列，y 是行
                piece = board.get_piece_at(pos)
                
                if piece is not None:
                    # 获取棋子对应的通道索引
                    channel_key = (piece.side, piece.type)
                    if channel_key in self.piece_to_channel:
                        channel_idx = self.piece_to_channel[channel_key]
                        encoded[channel_idx, row, col] = 1.0
        
        # 如果当前玩家是黑方，需要翻转棋盘视角
        if current_player == "black":
            encoded = self._flip_board_perspective(encoded)
            
        return encoded
    
    def _flip_board_perspective(self, encoded: np.ndarray) -> np.ndarray:
        """
        翻转棋盘视角（黑方视角）
        
        Args:
            encoded: 原始编码张量
            
        Returns:
            翻转后的编码张量
        """
        # 上下翻转棋盘
        flipped = np.flip(encoded, axis=1)
        
        # 交换红黑双方的通道
        result = np.zeros_like(flipped)
        
        # 红方通道 (0-6) 与黑方通道 (7-13) 交换
        result[0:7] = flipped[7:14]  # 黑方 -> 红方位置
        result[7:14] = flipped[0:7]  # 红方 -> 黑方位置
        
        return result
    
    def decode_board(self, encoded: np.ndarray, current_player: str) -> List[Tuple[Position, PieceType, str]]:
        """
        将编码张量解码为棋子位置列表
        
        Args:
            encoded: 编码张量
            current_player: 当前玩家颜色 ('red' 或 'black')
            
        Returns:
            棋子位置列表 [(位置, 棋子类型, 棋子颜色), ...]
        """
        pieces = []
        
        # 如果是黑方视角，先翻转回来
        if current_player == "black":
            encoded = self._flip_board_perspective(encoded)
        
        # 反向映射：通道索引到棋子类型
        channel_to_piece = {v: k for k, v in self.piece_to_channel.items()}
        
        # 遍历所有通道
        for channel_idx in range(14):
            if channel_idx in channel_to_piece:
                color, piece_type = channel_to_piece[channel_idx]
                
                # 找到该通道中值为1的位置
                positions = np.where(encoded[channel_idx] == 1.0)
                for row, col in zip(positions[0], positions[1]):
                    pos = Position(int(row), int(col))
                    pieces.append((pos, piece_type, color))
        
        return pieces


class ActionEncoder:
    """动作编码器"""
    
    def __init__(self):
        """初始化动作编码器"""
        # 中国象棋的动作空间：90个位置 * 90个目标位置 = 8100
        self.action_space_size = 90 * 90
        
    def encode_action(self, from_pos: Position, to_pos: Position) -> int:
        """
        将移动编码为动作索引
        
        Args:
            from_pos: 起始位置
            to_pos: 目标位置
            
        Returns:
            动作索引 (0 到 action_space_size-1)
        """
        # 使用Position的x, y属性而不是row, col
        from_idx = from_pos.y * 9 + from_pos.x  # 0-89
        to_idx = to_pos.y * 9 + to_pos.x      # 0-89
        
        # 动作编码：from_idx * 90 + to_idx
        action_idx = from_idx * 90 + to_idx
        
        # 确保索引在有效范围内
        if action_idx >= self.action_space_size:
            raise ValueError(f"动作索引 {action_idx} 超出范围 [0, {self.action_space_size})")
            
        return action_idx
    
    def decode_action(self, action_idx: int) -> Tuple[Position, Position]:
        """
        将动作索引解码为移动
        
        Args:
            action_idx: 动作索引
            
        Returns:
            (起始位置, 目标位置)
        """
        if not (0 <= action_idx < self.action_space_size):
            raise ValueError(f"动作索引 {action_idx} 超出范围 [0, {self.action_space_size})")
        
        # 计算起始位置索引
        from_idx = action_idx // 89
        to_offset = action_idx % 89
        
        # 计算目标位置索引（考虑跳过自移动）
        to_idx = to_offset if to_offset < from_idx else to_offset + 1
        
        # 将一维索引转换为二维位置
        from_y = from_idx // 9
        from_x = from_idx % 9
        to_y = to_idx // 9
        to_x = to_idx % 9
        
        return Position(from_x, from_y), Position(to_x, to_y)
    
    def encode_move_list(self, moves: List[Tuple[Position, Position]]) -> List[int]:
        """
        批量编码移动列表
        
        Args:
            moves: 移动列表 [(起始位置, 目标位置), ...]
            
        Returns:
            动作索引列表
        """
        return [self.encode_action(from_pos, to_pos) for from_pos, to_pos in moves]
    
    def decode_move_list(self, action_indices: List[int]) -> List[Tuple[Position, Position]]:
        """
        批量解码动作索引列表
        
        Args:
            action_indices: 动作索引列表
            
        Returns:
            移动列表 [(起始位置, 目标位置), ...]
        """
        return [self.decode_action(idx) for idx in action_indices]
    
    def create_action_mask(self, valid_moves: List[Tuple[Position, Position]]) -> np.ndarray:
        """
        创建合法动作掩码
        
        Args:
            valid_moves: 合法移动列表
            
        Returns:
            动作掩码，形状为 (action_space_size,)，合法动作为1，非法动作为0
        """
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        
        for from_pos, to_pos in valid_moves:
            try:
                action_idx = self.encode_action(from_pos, to_pos)
                mask[action_idx] = 1.0
            except ValueError:
                # 跳过无效的动作
                continue
                
        return mask


def test_encoders():
    """测试编码器功能"""
    from chess.core.game_state import GameState
    
    # 创建游戏状态
    game = GameState()
    
    # 测试棋盘编码
    board_encoder = BoardEncoder()
    encoded_board = board_encoder.encode_board(game.board, game.current_player)
    
    print(f"编码后的棋盘形状: {encoded_board.shape}")
    print(f"非零元素数量: {np.count_nonzero(encoded_board)}")
    
    # 测试动作编码
    action_encoder = ActionEncoder()
    
    # 获取合法移动
    valid_moves = game.get_valid_moves()
    print(f"合法移动数量: {len(valid_moves)}")
    
    if valid_moves:
        # 测试第一个移动的编码解码
        first_move = valid_moves[0]
        from_pos, to_pos = first_move
        
        action_idx = action_encoder.encode_action(from_pos, to_pos)
        decoded_move = action_encoder.decode_action(action_idx)
        
        print(f"原始移动: {from_pos} -> {to_pos}")
        print(f"动作索引: {action_idx}")
        print(f"解码移动: {decoded_move[0]} -> {decoded_move[1]}")
        print(f"编码解码一致: {first_move == decoded_move}")
        
        # 测试动作掩码
        action_mask = action_encoder.create_action_mask(valid_moves)
        print(f"动作掩码形状: {action_mask.shape}")
        print(f"合法动作数量: {np.sum(action_mask)}")


if __name__ == "__main__":
    test_encoders()