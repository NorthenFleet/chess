#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国象棋 PyQt GUI 界面
支持鼠标点击交互走棋
"""

import sys
import os
from typing import Optional, Tuple, List
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QMessageBox, 
                             QFrame, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap, QPalette

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.game_state import GameState, Player
from board.board import Board, Position
from piece.piece import Piece, PieceType
from rule.rule import Rule, MoveResult
from ai.network import ChessNet, create_chess_net
from ai.encoder import BoardEncoder, ActionEncoder
from ai.mcts import MCTS
from config import NetworkConfig, MCTSConfig


class ChessBoardWidget(QWidget):
    """象棋棋盘绘制组件"""
    
    # 信号：棋子被点击
    piece_clicked = pyqtSignal(int, int)  # row, col
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 700)
        self.setMaximumSize(800, 900)
        
        # 棋盘参数
        self.board_margin = 50
        self.cell_size = 60
        self.board_width = 8 * self.cell_size
        self.board_height = 9 * self.cell_size
        
        # 游戏状态
        self.game_state = None
        self.selected_pos = None  # 当前选中的位置
        self.valid_moves = []     # 当前选中棋子的合法移动
        
        # 设置背景色
        self.setStyleSheet("background-color: #F5DEB3;")  # 米色背景
        
    def set_game_state(self, game_state: GameState):
        """设置游戏状态"""
        self.game_state = game_state
        self.update()
        
    def set_selected_position(self, pos: Optional[Position]):
        """设置选中的位置"""
        self.selected_pos = pos
        if pos and self.game_state:
            # 获取该位置棋子的合法移动
            self.valid_moves = self._get_valid_moves(pos)
        else:
            self.valid_moves = []
        self.update()
        
    def _get_valid_moves(self, pos: Position) -> List[Position]:
        """获取指定位置棋子的所有合法移动"""
        if not self.game_state:
            return []
            
        valid_moves = []
        piece = self.game_state.board.get_piece_at(pos)
        if not piece:
            return []
            
        # 遍历所有可能的目标位置
        for row in range(10):
            for col in range(9):
                target_pos = Position(col, row)  # x=col, y=row
                if self.game_state.rule.validate_move(self.game_state, pos, target_pos).is_valid:
                    valid_moves.append(target_pos)
                    
        return valid_moves
        
    def paintEvent(self, event):
        """绘制棋盘"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制棋盘背景
        self._draw_board(painter)
        
        # 绘制棋子
        if self.game_state:
            self._draw_pieces(painter)
            
        # 绘制选中状态和合法移动提示
        self._draw_selection_hints(painter)
        
    def _draw_board(self, painter: QPainter):
        """绘制棋盘线条"""
        pen = QPen(QColor(0, 0, 0), 2)
        painter.setPen(pen)
        
        # 计算棋盘起始位置
        start_x = self.board_margin
        start_y = self.board_margin
        
        # 绘制横线
        for i in range(10):
            y = start_y + i * self.cell_size
            painter.drawLine(start_x, y, start_x + self.board_width, y)
            
        # 绘制竖线
        for i in range(9):
            x = start_x + i * self.cell_size
            # 上半部分
            painter.drawLine(x, start_y, x, start_y + 4 * self.cell_size)
            # 下半部分
            painter.drawLine(x, start_y + 5 * self.cell_size, x, start_y + 9 * self.cell_size)
            
        # 绘制九宫格对角线
        # 上方九宫格
        painter.drawLine(start_x + 3 * self.cell_size, start_y,
                        start_x + 5 * self.cell_size, start_y + 2 * self.cell_size)
        painter.drawLine(start_x + 5 * self.cell_size, start_y,
                        start_x + 3 * self.cell_size, start_y + 2 * self.cell_size)
        
        # 下方九宫格
        painter.drawLine(start_x + 3 * self.cell_size, start_y + 7 * self.cell_size,
                        start_x + 5 * self.cell_size, start_y + 9 * self.cell_size)
        painter.drawLine(start_x + 5 * self.cell_size, start_y + 7 * self.cell_size,
                        start_x + 3 * self.cell_size, start_y + 9 * self.cell_size)
                        
        # 绘制楚河汉界
        font = QFont("SimHei", 16, QFont.Bold)
        painter.setFont(font)
        painter.drawText(int(start_x + self.cell_size), int(start_y + 4.7 * self.cell_size), "楚河")
        painter.drawText(int(start_x + 5 * self.cell_size), int(start_y + 4.7 * self.cell_size), "汉界")
        
    def _draw_pieces(self, painter: QPainter):
        """绘制棋子"""
        if not self.game_state:
            return
            
        # 设置棋子样式
        piece_radius = 25
        font = QFont("SimHei", 14, QFont.Bold)
        painter.setFont(font)
        
        for row in range(10):
            for col in range(9):
                pos = Position(col, row)  # x=col, y=row
                piece = self.game_state.board.get_piece_at(pos)
                if piece:
                    # 计算棋子中心位置
                    x = self.board_margin + col * self.cell_size
                    y = self.board_margin + row * self.cell_size
                    
                    # 设置棋子颜色
                    if piece.side == 'red':
                        brush = QBrush(QColor(220, 20, 20))  # 红色
                        text_color = QColor(255, 255, 255)   # 白色文字
                    else:
                        brush = QBrush(QColor(20, 20, 20))   # 黑色
                        text_color = QColor(255, 255, 255)   # 白色文字
                        
                    # 绘制棋子圆形背景
                    painter.setBrush(brush)
                    painter.setPen(QPen(QColor(0, 0, 0), 2))
                    painter.drawEllipse(x - piece_radius, y - piece_radius, 
                                      piece_radius * 2, piece_radius * 2)
                    
                    # 绘制棋子文字
                    painter.setPen(QPen(text_color))
                    piece_text = self._get_piece_text(piece)
                    text_rect = QRect(x - piece_radius, y - piece_radius,
                                    piece_radius * 2, piece_radius * 2)
                    painter.drawText(text_rect, Qt.AlignCenter, piece_text)
                    
    def _get_piece_text(self, piece: Piece) -> str:
        """获取棋子显示文字"""
        piece_names = {
            PieceType.GENERAL: "帅" if piece.side == 'red' else "将",
            PieceType.ADVISOR: "仕" if piece.side == 'red' else "士", 
            PieceType.ELEPHANT: "相" if piece.side == 'red' else "象",
            PieceType.HORSE: "马",
            PieceType.CHARIOT: "车",
            PieceType.CANNON: "炮" if piece.side == 'red' else "砲",
            PieceType.SOLDIER: "兵" if piece.side == 'red' else "卒"
        }
        return piece_names.get(piece.type, "?")
        
    def _draw_selection_hints(self, painter: QPainter):
        """绘制选中状态和移动提示"""
        if not self.selected_pos:
            return
            
        # 绘制选中棋子的高亮
        x = self.board_margin + self.selected_pos.x * self.cell_size
        y = self.board_margin + self.selected_pos.y * self.cell_size
        
        painter.setPen(QPen(QColor(255, 255, 0), 4))  # 黄色高亮
        painter.setBrush(QBrush(Qt.NoBrush))
        painter.drawEllipse(x - 30, y - 30, 60, 60)
        
        # 绘制合法移动位置的提示
        painter.setPen(QPen(QColor(0, 255, 0), 2))  # 绿色提示
        painter.setBrush(QBrush(QColor(0, 255, 0, 100)))
        
        for move_pos in self.valid_moves:
            x = self.board_margin + move_pos.x * self.cell_size
            y = self.board_margin + move_pos.y * self.cell_size
            painter.drawEllipse(x - 8, y - 8, 16, 16)
            
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            # 计算点击的棋盘位置
            x = event.x() - self.board_margin
            y = event.y() - self.board_margin
            
            if 0 <= x <= self.board_width and 0 <= y <= self.board_height:
                col = round(x / self.cell_size)
                row = round(y / self.cell_size)
                
                if 0 <= row <= 9 and 0 <= col <= 8:
                    clicked_pos = Position(col, row)  # x=col, y=row
                    self.piece_clicked.emit(row, col)


class ChessMainWindow(QMainWindow):
    """象棋游戏主窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_game()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("中国象棋 - PyQt版")
        self.setGeometry(100, 100, 900, 800)
        
        # 创建中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建棋盘组件
        self.board_widget = ChessBoardWidget()
        self.board_widget.piece_clicked.connect(self.on_piece_clicked)
        main_layout.addWidget(self.board_widget)
        
        # 创建右侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(250)
        
        layout = QVBoxLayout(panel)
        
        # 游戏状态显示
        self.status_label = QLabel("游戏状态: 准备开始")
        self.status_label.setFont(QFont("SimHei", 12))
        layout.addWidget(self.status_label)
        
        # 当前玩家显示
        self.player_label = QLabel("当前玩家: 红方")
        self.player_label.setFont(QFont("SimHei", 12))
        layout.addWidget(self.player_label)
        
        # 控制按钮
        self.new_game_btn = QPushButton("新游戏")
        self.new_game_btn.clicked.connect(self.new_game)
        layout.addWidget(self.new_game_btn)
        
        self.ai_move_btn = QPushButton("AI走棋")
        self.ai_move_btn.clicked.connect(self.ai_move)
        layout.addWidget(self.ai_move_btn)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
        
    def init_game(self):
        """初始化游戏"""
        self.game_state = GameState()
        self.game_state.setup_players("人类玩家", "AI玩家")
        self.game_state.start_game()
        
        # 初始化AI（可选）
        self.ai_player = None
        try:
            from human_vs_ai import AIPlayer
            self.ai_player = AIPlayer()
        except Exception as e:
            print(f"AI初始化失败: {e}")
            
        # 更新界面
        self.board_widget.set_game_state(self.game_state)
        self.update_status()
        
    def on_piece_clicked(self, row: int, col: int):
        """处理棋子点击事件"""
        clicked_pos = Position(col, row)  # x=col, y=row
        
        # 如果已经选中了棋子
        if self.board_widget.selected_pos:
            selected_pos = self.board_widget.selected_pos
            
            # 如果点击的是同一个位置，取消选择
            if clicked_pos == selected_pos:
                self.board_widget.set_selected_position(None)
                return
                
            # 尝试移动棋子
            if self.try_move_piece(selected_pos, clicked_pos):
                self.board_widget.set_selected_position(None)
                self.update_status()
                
                # 检查游戏是否结束
                if self.game_state.is_game_over():
                    self.show_game_over()
            else:
                # 如果移动失败，检查点击的位置是否有己方棋子
                piece = self.game_state.board.get_piece_at(clicked_pos)
                if piece and piece.side == self.game_state.current_player.side:
                    self.board_widget.set_selected_position(clicked_pos)
                else:
                    self.board_widget.set_selected_position(None)
        else:
            # 没有选中棋子，检查点击位置是否有己方棋子
            piece = self.game_state.board.get_piece_at(clicked_pos)
            if piece and piece.side == self.game_state.current_player.side:
                self.board_widget.set_selected_position(clicked_pos)
                
    def try_move_piece(self, from_pos: Position, to_pos: Position) -> bool:
        """尝试移动棋子"""
        return self.game_state.make_move(from_pos, to_pos)
        
    def update_status(self):
        """更新游戏状态显示"""
        if self.game_state.is_game_over():
            winner = self.game_state.get_winner()
            if winner:
                winner_text = "红方" if winner == 'red' else "黑方"
                self.status_label.setText(f"游戏结束 - {winner_text}获胜!")
            else:
                self.status_label.setText("游戏结束 - 平局!")
        else:
            self.status_label.setText("游戏进行中")
            
        current_player_text = "红方" if self.game_state.current_player.side == 'red' else "黑方"
        self.player_label.setText(f"当前玩家: {current_player_text}")
        
    def new_game(self):
        """开始新游戏"""
        self.init_game()
        
    def ai_move(self):
        """AI走棋"""
        if not self.ai_player or self.game_state.is_game_over():
            return
            
        try:
            from_pos, to_pos = self.ai_player.get_move(self.game_state)
            if from_pos and to_pos:
                if self.game_state.make_move(from_pos, to_pos):
                    self.board_widget.set_game_state(self.game_state)
                    self.update_status()
                    
                    if self.game_state.is_game_over():
                        self.show_game_over()
        except Exception as e:
            QMessageBox.warning(self, "AI错误", f"AI走棋失败: {e}")
            
    def show_game_over(self):
        """显示游戏结束对话框"""
        winner = self.game_state.get_winner()
        if winner:
            winner_text = "红方" if winner == 'red' else "黑方"
            message = f"游戏结束！\n{winner_text}获胜！"
        else:
            message = "游戏结束！\n平局！"
            
        reply = QMessageBox.question(self, "游戏结束", 
                                   message + "\n\n是否开始新游戏？",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.new_game()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("中国象棋")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = ChessMainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()