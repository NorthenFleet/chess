"""
中国象棋棋子组件
支持拖拽功能
"""

from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QMimeData, QSize, QPropertyAnimation, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QDrag, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from typing import Dict, Optional

from ..assets.chess_pieces import get_piece_svg

class ChessPieceWidget(QLabel):
    """中国象棋棋子组件"""
    
    # 信号
    piece_selected = pyqtSignal(object)  # 棋子被选中
    piece_moved = pyqtSignal(object, QPoint)  # 棋子被移动
    drag_started = pyqtSignal(object)  # 开始拖拽
    drag_finished = pyqtSignal(object)  # 拖拽结束
    
    def __init__(self, piece_info: Dict, size: int = 50, parent=None):
        super().__init__(parent)
        
        self.piece_info = piece_info
        self.piece_size = size
        self.drag_start_position = QPoint()
        self.is_dragging = False
        
        # 设置组件属性
        self.setFixedSize(QSize(size, size))
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
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
        
        # 创建棋子图像
        self._create_piece_pixmap()
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)
    
    def _create_piece_pixmap(self):
        """创建棋子图像"""
        # 获取棋子信息
        piece_type = self.piece_info.get('type', '')
        piece_color = self.piece_info.get('color', 'red')
        
        if not piece_type or not piece_color:
            pixmap = QPixmap(self.piece_size, self.piece_size)
            pixmap.fill(Qt.transparent)
            self.setPixmap(pixmap)
            return
        
        # 获取SVG图像
        svg_data = get_piece_svg(piece_type, piece_color)
        if svg_data:
            # 渲染SVG
            pixmap = QPixmap(self.piece_size, self.piece_size)
            pixmap.fill(Qt.transparent)
            
            renderer = QSvgRenderer()
            renderer.load(svg_data.encode('utf-8'))
            
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            
            self.setPixmap(pixmap)
        else:
            # 如果没有SVG，创建简单的文字棋子
            self._create_text_piece()
    
    def _create_text_piece(self):
        """创建文字棋子（备用方案）"""
        pixmap = QPixmap(self.piece_size, self.piece_size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取棋子信息
        piece_type = self.piece_info.get('type', '')
        piece_color = self.piece_info.get('color', 'red')
        
        # 设置颜色
        if piece_color == 'red':
            bg_color = QColor(255, 200, 200)
            text_color = QColor(200, 0, 0)
            border_color = QColor(150, 0, 0)
        else:
            bg_color = QColor(200, 200, 200)
            text_color = QColor(0, 0, 0)
            border_color = QColor(50, 50, 50)
        
        # 绘制棋子背景圆圈
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 2))
        painter.drawEllipse(2, 2, self.piece_size - 4, self.piece_size - 4)
        
        # 绘制棋子文字
        font = QFont("SimHei", max(12, self.piece_size // 3), QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(text_color, 2))
        
        piece_text = self.piece_names.get(piece_color, {}).get(piece_type, '?')
        
        # 计算文字居中位置
        fm = painter.fontMetrics()
        text_width = fm.width(piece_text)
        text_height = fm.height()
        
        text_x = (self.piece_size - text_width) // 2
        text_y = (self.piece_size + text_height // 2) // 2
        
        painter.drawText(text_x, text_y, piece_text)
        painter.end()
        
        self.setPixmap(pixmap)
    
    def get_piece_info(self) -> Dict:
        """获取棋子信息"""
        return self.piece_info.copy()
    
    def set_piece_info(self, piece_info: Dict):
        """设置棋子信息"""
        self.piece_info = piece_info
        self._create_piece_pixmap()
    
    def get_piece_type(self) -> str:
        """获取棋子类型"""
        return self.piece_info.get('type', '')
    
    def get_piece_color(self) -> str:
        """获取棋子颜色"""
        return self.piece_info.get('color', 'red')
    
    def get_piece_position(self) -> tuple:
        """获取棋子位置"""
        return self.piece_info.get('position', (0, 0))
    
    def set_piece_position(self, row: int, col: int):
        """设置棋子位置"""
        self.piece_info['position'] = (row, col)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()
            self.piece_selected.emit(self)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if not (event.buttons() & Qt.LeftButton):
            return
        
        if ((event.pos() - self.drag_start_position).manhattanLength() < 
            Qt.QApplication.startDragDistance()):
            return
        
        # 开始拖拽
        if not self.is_dragging:
            self.is_dragging = True
            self.drag_started.emit(self)
            self._start_drag()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.is_dragging:
            self.is_dragging = False
            self.drag_finished.emit(self)
            self.piece_moved.emit(self, event.globalPos())
    
    def _start_drag(self):
        """开始拖拽操作"""
        drag = QDrag(self)
        mime_data = QMimeData()
        
        # 设置拖拽数据
        piece_data = f"{self.get_piece_color()}_{self.get_piece_type()}"
        mime_data.setText(piece_data)
        
        # 设置拖拽时的图像
        pixmap = self.pixmap()
        if pixmap:
            # 创建半透明的拖拽图像
            drag_pixmap = QPixmap(pixmap.size())
            drag_pixmap.fill(Qt.transparent)
            
            painter = QPainter(drag_pixmap)
            painter.setOpacity(0.7)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()
            
            drag.setPixmap(drag_pixmap)
            drag.setHotSpot(QPoint(self.piece_size // 2, self.piece_size // 2))
        
        drag.setMimeData(mime_data)
        
        # 执行拖拽
        drop_action = drag.exec_(Qt.MoveAction)
        
        return drop_action
    
    def enterEvent(self, event):
        """鼠标进入事件"""
        self.setCursor(Qt.PointingHandCursor)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)
    
    def set_highlighted(self, highlighted: bool):
        """设置高亮状态"""
        if highlighted:
            self.setStyleSheet("""
                QLabel {
                    border: 3px solid yellow;
                    border-radius: 25px;
                    background-color: rgba(255, 255, 0, 50);
                }
            """)
        else:
            self.setStyleSheet("")
    
    def set_selectable(self, selectable: bool):
        """设置是否可选择"""
        if selectable:
            self.setCursor(Qt.PointingHandCursor)
            self.setEnabled(True)
        else:
            self.setCursor(Qt.ForbiddenCursor)
            self.setEnabled(False)
    
    def animate_move(self, from_pos: QPoint, to_pos: QPoint, duration: int = 300):
        """动画移动棋子"""
        from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
        
        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setDuration(duration)
        self.animation.setStartValue(from_pos)
        self.animation.setEndValue(to_pos)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()
    
    def __str__(self):
        """字符串表示"""
        color = self.get_piece_color()
        piece_type = self.get_piece_type()
        position = self.get_piece_position()
        return f"{color}_{piece_type}_at_{position}"
    
    def __repr__(self):
        """对象表示"""
        return f"ChessPieceWidget({self.piece_info})"