import copy
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from rule.rule import MoveResult


class EventType(Enum):
    """游戏事件类型枚举"""
    GAME_START = "game_start"  # 游戏开始
    GAME_END = "game_end"      # 游戏结束
    MOVE = "move"              # 移动棋子
    CAPTURE = "capture"        # 吃子
    CHECK = "check"            # 将军
    CHECKMATE = "checkmate"    # 将死
    DRAW = "draw"              # 和棋
    TURN_CHANGE = "turn_change"  # 回合变更


@dataclass
class GameEvent:
    """游戏事件数据类"""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Player:
    """玩家类"""
    def __init__(self, name: str, side: str):
        """
        初始化玩家
        
        Args:
            name: 玩家名称
            side: 玩家方（'red' 或 'black'）
        """
        self.name = name
        self.side = side  # 'red' 或 'black'
        self.captured_pieces = []  # 被吃的棋子
        
    def __str__(self) -> str:
        return f"{self.name} ({self.side})"


class GameState:
    """游戏状态类，管理游戏的核心状态和事件系统"""
    
    def __init__(self):
        """初始化游戏状态"""
        self.current_player = None  # 当前回合的玩家
        self.players = {}  # 玩家字典，键为 'red' 或 'black'
        self.turn_number = 0  # 回合数
        self.game_over = False  # 游戏是否结束
        self.winner = None  # 获胜者
        self.start_time = None  # 游戏开始时间
        self.end_time = None  # 游戏结束时间
        
        # 初始化棋盘和规则
        from board.board import Board
        from rule.rule import Rule
        self.board = Board()
        self.rule = Rule(self.board)
        
        # 事件系统
        self.event_listeners: Dict[EventType, List[Callable]] = {}
        for event_type in EventType:
            self.event_listeners[event_type] = []
        
        # 游戏历史记录
        self.history: List[GameEvent] = []
    
    def setup_players(self, red_player_name: str, black_player_name: str) -> None:
        """设置玩家"""
        self.players['red'] = Player(red_player_name, 'red')
        self.players['black'] = Player(black_player_name, 'black')
        self.current_player = self.players['red']  # 红方先行
    
    def start_game(self) -> None:
        """开始游戏"""
        if not self.players or len(self.players) != 2:
            raise ValueError("游戏需要两名玩家才能开始")
            
        self.start_time = datetime.now()
        self.turn_number = 1
        self.game_over = False
        self.winner = None
        
        # 初始化棋盘
        self.rule.initialize_board()
        
        # 触发游戏开始事件
        self.trigger_event(EventType.GAME_START, {
            'players': self.players,
            'start_time': self.start_time
        })
    
    def end_game(self, winner: Optional[Player] = None, reason: str = "") -> None:
        """结束游戏"""
        self.game_over = True
        self.winner = winner
        self.end_time = datetime.now()
        
        # 触发游戏结束事件
        self.trigger_event(EventType.GAME_END, {
            'winner': winner.name if winner else None,
            'reason': reason,
            'end_time': self.end_time,
            'duration': (self.end_time - self.start_time).total_seconds() if self.start_time else None
        })
    
    def switch_turn(self) -> None:
        """切换回合"""
        if self.game_over:
            return
            
        # 切换当前玩家
        self.current_player = self.players['black'] if self.current_player == self.players['red'] else self.players['red']
        self.turn_number += 1
        
        # 触发回合变更事件
        self.trigger_event(EventType.TURN_CHANGE, {
            'player': self.current_player.name,
            'side': self.current_player.side,
            'turn_number': self.turn_number
        })
    
    def make_move(self, from_pos, to_pos) -> bool:
        """执行移动"""
        if not self.current_player:
            return False
        
        # 使用Rule的execute_move方法
        result, captured_piece = self.rule.execute_move(from_pos, to_pos)
        
        if result == MoveResult.INVALID:
            return False
        
        # 检查游戏是否结束
        if result in [MoveResult.CHECKMATE, MoveResult.STALEMATE]:
            self.game_over = True
            # 设置获胜者
            if result == MoveResult.CHECKMATE:
                # 当前玩家获胜（因为对手被将死）
                self.winner = self.current_player.side
            else:
                # 和棋
                self.winner = None
        
        # 切换玩家
        self.switch_turn()
        
        return True
    
    def get_valid_moves(self):
        """获取当前玩家的所有合法移动"""
        if not self.current_player:
            return []
        
        valid_moves_dict = self.rule.get_all_valid_moves(self.current_player.side)
        # 将字典转换为(from_pos, to_pos)元组列表
        moves = []
        for from_pos, to_positions in valid_moves_dict.items():
            for to_pos in to_positions:
                moves.append((from_pos, to_pos))
        return moves
    
    def get_winner(self):
        """获取获胜者"""
        return self.winner.side if self.winner else None
    
    def register_event_listener(self, event_type: EventType, callback: Callable) -> None:
        """注册事件监听器"""
        if event_type in self.event_listeners:
            self.event_listeners[event_type].append(callback)
    
    def unregister_event_listener(self, event_type: EventType, callback: Callable) -> None:
        """注销事件监听器"""
        if event_type in self.event_listeners and callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
    
    def trigger_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """触发事件"""
        event = GameEvent(type=event_type, data=data)
        self.history.append(event)  # 记录到历史
        
        # 通知所有监听该事件类型的回调函数
        if event_type in self.event_listeners:
            for callback in self.event_listeners[event_type]:
                callback(event)
    
    def get_game_summary(self) -> Dict[str, Any]:
        """获取游戏摘要"""
        return {
            'game_over': self.game_over,
            'winner': self.winner.name if self.winner else None,
            'turn_number': self.turn_number,
            'current_player': self.current_player.name if self.current_player else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            'red_player': self.players.get('red').name if 'red' in self.players else None,
            'black_player': self.players.get('black').name if 'black' in self.players else None,
        }
    
    def copy(self):
        """创建GameState的深拷贝"""
        return copy.deepcopy(self)