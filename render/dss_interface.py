"""
DSS (Data Service System) 接口
用于前端与游戏引擎之间的通信
"""

import json
from typing import Dict, Any, Optional, Callable
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from enum import Enum

class MessageType(Enum):
    """消息类型枚举"""
    GAME_STATE = "game_state"
    MOVE_REQUEST = "move_request"
    MOVE_RESPONSE = "move_response"
    GAME_OVER = "game_over"
    ERROR = "error"
    RESET_GAME = "reset_game"
    AI_THINKING = "ai_thinking"

class DSSInterface(QObject):
    """DSS通信接口类"""
    
    # PyQt信号
    game_state_updated = pyqtSignal(dict)
    move_completed = pyqtSignal(dict)
    game_over = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    ai_thinking_status = pyqtSignal(bool)
    
    def __init__(self, game_engine=None):
        super().__init__()
        self.game_engine = game_engine
        self.message_handlers = {}
        self.setup_handlers()
        
        # 定时器用于轮询游戏状态
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self.poll_game_state)
        self.poll_timer.start(100)  # 100ms轮询间隔
    
    def setup_handlers(self):
        """设置消息处理器"""
        self.message_handlers = {
            MessageType.GAME_STATE: self._handle_game_state,
            MessageType.MOVE_RESPONSE: self._handle_move_response,
            MessageType.GAME_OVER: self._handle_game_over,
            MessageType.ERROR: self._handle_error,
            MessageType.AI_THINKING: self._handle_ai_thinking
        }
    
    def set_game_engine(self, game_engine):
        """设置游戏引擎实例"""
        self.game_engine = game_engine
    
    def send_message(self, message_type: MessageType, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """发送消息到游戏引擎"""
        if not self.game_engine:
            self.error_occurred.emit("游戏引擎未初始化")
            return None
        
        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": self._get_timestamp()
        }
        
        try:
            # 根据消息类型调用相应的游戏引擎方法
            if message_type == MessageType.MOVE_REQUEST:
                return self._handle_move_request(data)
            elif message_type == MessageType.RESET_GAME:
                return self._handle_reset_game(data)
            else:
                return self._handle_generic_request(message)
        except Exception as e:
            self.error_occurred.emit(f"发送消息失败: {str(e)}")
            return None
    
    def _handle_move_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理移动请求"""
        try:
            from_pos = data.get('from_pos')
            to_pos = data.get('to_pos')
            
            if not from_pos or not to_pos:
                raise ValueError("移动位置信息不完整")
            
            # 调用游戏引擎的移动方法
            if hasattr(self.game_engine, 'make_move'):
                result = self.game_engine.make_move(from_pos, to_pos)
                return {
                    "success": result,
                    "message": "移动成功" if result else "移动失败"
                }
            else:
                raise AttributeError("游戏引擎不支持make_move方法")
                
        except Exception as e:
            return {
                "success": False,
                "message": f"移动处理失败: {str(e)}"
            }
    
    def _handle_reset_game(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理重置游戏请求"""
        try:
            if hasattr(self.game_engine, 'reset_game'):
                self.game_engine.reset_game()
                return {
                    "success": True,
                    "message": "游戏重置成功"
                }
            else:
                raise AttributeError("游戏引擎不支持reset_game方法")
        except Exception as e:
            return {
                "success": False,
                "message": f"游戏重置失败: {str(e)}"
            }
    
    def _handle_generic_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用请求"""
        # 这里可以扩展其他类型的请求处理
        return {
            "success": True,
            "message": "请求已接收"
        }
    
    def poll_game_state(self):
        """轮询游戏状态"""
        if not self.game_engine:
            return
        
        try:
            # 获取当前游戏状态
            if hasattr(self.game_engine, 'get_game_state'):
                state = self.game_engine.get_game_state()
                if state:
                    self.game_state_updated.emit(state)
            
            # 检查游戏是否结束
            if hasattr(self.game_engine, 'is_game_over'):
                if self.game_engine.is_game_over():
                    game_result = getattr(self.game_engine, 'get_game_result', lambda: {})()
                    self.game_over.emit(game_result)
                    
        except Exception as e:
            self.error_occurred.emit(f"轮询游戏状态失败: {str(e)}")
    
    def _handle_game_state(self, data: Dict[str, Any]):
        """处理游戏状态消息"""
        self.game_state_updated.emit(data)
    
    def _handle_move_response(self, data: Dict[str, Any]):
        """处理移动响应消息"""
        self.move_completed.emit(data)
    
    def _handle_game_over(self, data: Dict[str, Any]):
        """处理游戏结束消息"""
        self.game_over.emit(data)
    
    def _handle_error(self, data: Dict[str, Any]):
        """处理错误消息"""
        error_msg = data.get('message', '未知错误')
        self.error_occurred.emit(error_msg)
    
    def _handle_ai_thinking(self, data: Dict[str, Any]):
        """处理AI思考状态消息"""
        is_thinking = data.get('thinking', False)
        self.ai_thinking_status.emit(is_thinking)
    
    def _get_timestamp(self) -> float:
        """获取当前时间戳"""
        import time
        return time.time()
    
    def stop_polling(self):
        """停止轮询"""
        if self.poll_timer.isActive():
            self.poll_timer.stop()
    
    def start_polling(self):
        """开始轮询"""
        if not self.poll_timer.isActive():
            self.poll_timer.start(100)