"""
中国象棋棋盘显示组件
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap
from typing import Dict, Tuple, Optional, List

class ChessBoardWidget(QWidget):
    """中国象棋棋盘组件"""
    
    # 信号
    piece_clicked = pyqtSignal(int, int)  # 棋子被点击
    square_clicked = pyqtSignal(int, int)  # 棋盘格子被点击
    move_requested = pyqtSignal(tuple, tuple)  # 请求移动棋子
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 700)
        self.setMaximumSize(800, 900)
        
        # 棋盘参数
        self.board_margin = 50
        self.grid_size = 60
        self.piece_size = 50
        
        # 棋盘状态
        self.board_state = {}  # 存储棋盘状态 {(row, col): piece_info}
        self.selected_pos = None  # 当前选中的位置
        self.valid_moves = []  # 当前选中棋子的有效移动
        self.last_move = None  # 上一步移动
        
        # 颜色定义
        self.board_color = QColor(245, 222, 179)  # 棋盘背景色
        self.line_color = QColor(0, 0, 0)  # 线条颜色
        self.selected_color = QColor(255, 255, 0, 100)  # 选中高亮色
        self.valid_move_color = QColor(0, 255, 0, 100)  # 有效移动高亮色
        self.last_move_color = QColor(255, 0, 0, 100)  # 上一步移动高亮色
        
        # 中文棋子名称映射
        self.piece_names = {
            'red': {
                'king': '帅', 'advisor': '仕', 'elephant': '相',
                'horse': '马', 'rook': '车', 'cannon': '炮', 'pawn': '兵'
            },
            'black': {
                'king': '将', 'advisor': '士', 'elephant': '象',
                'horse': '马', 'rook': '车', 'cannon': '炮', 'pawn': '卒'
            }
        }
        
        self.setMouseTracking(True)
    
    def set_board_state(self, board_state: Dict):
        """设置棋盘状态"""
        self.board_state = board_state
        self.update()
    
    def set_selected_position(self, pos: Optional[Tuple[int, int]]):
        """设置选中位置"""
        self.selected_pos = pos
        self.update()
    
    def set_valid_moves(self, moves: List[Tuple[int, int]]):
        """设置有效移动位置"""
        self.valid_moves = moves
        self.update()
    
    def set_last_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """设置上一步移动"""
        self.last_move = (from_pos, to_pos)
        self.update()
    
    def paintEvent(self, event):
        """绘制棋盘"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), self.board_color)
        
        # 绘制棋盘网格
        self._draw_board_grid(painter)
        
        # 绘制特殊标记（九宫格、河界等）
        self._draw_special_marks(painter)
        
        # 绘制高亮
        self._draw_highlights(painter)
        
        # 绘制棋子
        self._draw_pieces(painter)
    
    def _draw_board_grid(self, painter: QPainter):
        """绘制棋盘网格"""
        pen = QPen(self.line_color, 2)
        painter.setPen(pen)
        
        # 计算棋盘区域
        board_width = 8 * self.grid_size
        board_height = 9 * self.grid_size
        
        start_x = self.board_margin
        start_y = self.board_margin
        
        # 绘制横线
        for i in range(10):
            y = start_y + i * self.grid_size
            painter.drawLine(start_x, y, start_x + board_width, y)
        
        # 绘制竖线
        for i in range(9):
            x = start_x + i * self.grid_size
            # 上半部分
            painter.drawLine(x, start_y, x, start_y + 4 * self.grid_size)
            # 下半部分
            painter.drawLine(x, start_y + 5 * self.grid_size, x, start_y + 9 * self.grid_size)
    
    def _draw_special_marks(self, painter: QPainter):
        """绘制特殊标记"""
        pen = QPen(self.line_color, 2)
        painter.setPen(pen)
        
        start_x = self.board_margin
        start_y = self.board_margin
        
        # 绘制九宫格对角线
        # 上方九宫格
        painter.drawLine(start_x + 3 * self.grid_size, start_y,
                        start_x + 5 * self.grid_size, start_y + 2 * self.grid_size)
        painter.drawLine(start_x + 5 * self.grid_size, start_y,
                        start_x + 3 * self.grid_size, start_y + 2 * self.grid_size)
        
        # 下方九宫格
        painter.drawLine(start_x + 3 * self.grid_size, start_y + 7 * self.grid_size,
                        start_x + 5 * self.grid_size, start_y + 9 * self.grid_size)
        painter.drawLine(start_x + 5 * self.grid_size, start_y + 7 * self.grid_size,
                        start_x + 3 * self.grid_size, start_y + 9 * self.grid_size)
        
        # 绘制河界文字
        font = QFont("SimHei", 16)
        painter.setFont(font)
        painter.setPen(QPen(QColor(139, 69, 19), 2))
        
        # "楚河"
        painter.drawText(int(start_x + 1 * self.grid_size), int(start_y + 4.7 * self.grid_size), "楚河")
        # "汉界"
        painter.drawText(int(start_x + 5 * self.grid_size), int(start_y + 4.7 * self.grid_size), "汉界")
    
    def _draw_highlights(self, painter: QPainter):
        """绘制高亮效果"""
        start_x = self.board_margin
        start_y = self.board_margin
        
        # 绘制上一步移动高亮
        if self.last_move:
            from_pos, to_pos = self.last_move
            brush = QBrush(self.last_move_color)
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)
            
            for pos in [from_pos, to_pos]:
                row, col = pos
                x = start_x + col * self.grid_size - self.piece_size // 2
                y = start_y + row * self.grid_size - self.piece_size // 2
                painter.drawEllipse(x, y, self.piece_size, self.piece_size)
        
        # 绘制选中位置高亮
        if self.selected_pos:
            row, col = self.selected_pos
            x = start_x + col * self.grid_size - self.piece_size // 2
            y = start_y + row * self.grid_size - self.piece_size // 2
            
            brush = QBrush(self.selected_color)
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x, y, self.piece_size, self.piece_size)
        
        # 绘制有效移动位置高亮
        if self.valid_moves:
            brush = QBrush(self.valid_move_color)
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)
            
            for row, col in self.valid_moves:
                x = start_x + col * self.grid_size - self.piece_size // 2
                y = start_y + row * self.grid_size - self.piece_size // 2
                painter.drawEllipse(x, y, self.piece_size, self.piece_size)
    
    def _draw_pieces(self, painter: QPainter):
        """绘制棋子"""
        start_x = self.board_margin
        start_y = self.board_margin
        
        for (row, col), piece_info in self.board_state.items():
            x = start_x + col * self.grid_size
            y = start_y + row * self.grid_size
            
            self._draw_single_piece(painter, x, y, piece_info)
    
    def _draw_single_piece(self, painter: QPainter, x: int, y: int, piece_info: Dict):
        """绘制单个棋子"""
        piece_type = piece_info.get('type', '')
        piece_color = piece_info.get('color', 'red')
        
        # 棋子背景圆圈
        if piece_color == 'red':
            bg_color = QColor(255, 200, 200)
            text_color = QColor(200, 0, 0)
        else:
            bg_color = QColor(200, 200, 200)
            text_color = QColor(0, 0, 0)
        
        # 绘制棋子背景
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawEllipse(x - self.piece_size // 2, y - self.piece_size // 2,
                          self.piece_size, self.piece_size)
        
        # 绘制棋子文字
        font = QFont("SimHei", 18, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(text_color, 2))
        
        piece_text = self.piece_names.get(piece_color, {}).get(piece_type, '?')
        
        # 计算文字居中位置
        fm = painter.fontMetrics()
        text_width = fm.width(piece_text)
        text_height = fm.height()
        
        text_x = x - text_width // 2
        text_y = y + text_height // 4
        
        painter.drawText(text_x, text_y, piece_text)
    
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            pos = self._get_board_position(event.pos())
            if pos:
                row, col = pos
                
                # 检查是否点击了棋子
                if pos in self.board_state:
                    self.piece_clicked.emit(row, col)
                else:
                    self.square_clicked.emit(row, col)
                
                # 如果有选中的棋子且点击了有效移动位置，发出移动请求
                if self.selected_pos and pos in self.valid_moves:
                    self.move_requested.emit(self.selected_pos, pos)
    
    def _get_board_position(self, point: QPoint) -> Optional[Tuple[int, int]]:
        """将屏幕坐标转换为棋盘位置"""
        x = point.x() - self.board_margin
        y = point.y() - self.board_margin
        
        # 检查是否在棋盘范围内
        if x < 0 or y < 0:
            return None
        
        col = round(x / self.grid_size)
        row = round(y / self.grid_size)
        
        # 检查位置是否有效
        if 0 <= row <= 9 and 0 <= col <= 8:
            return (row, col)
        
        return None
    
    def get_piece_at_position(self, row: int, col: int) -> Optional[Dict]:
        """获取指定位置的棋子信息"""
        return self.board_state.get((row, col))
    
    def clear_highlights(self):
        """清除所有高亮"""
        self.selected_pos = None
        self.valid_moves = []
        self.update()