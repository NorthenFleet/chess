"""
中国象棋游戏主窗口
"""

import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QTextEdit, QSplitter, QFrame,
                           QMenuBar, QMenu, QAction, QStatusBar, QMessageBox,
                           QGroupBox, QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QFont, QIcon, QPixmap

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from render.components.chess_board import ChessBoardWidget
from render.dss_interface import DSSInterface, MessageType

class ChessMainWindow(QMainWindow):
    """中国象棋游戏主窗口"""
    
    def __init__(self, game_engine=None, parent=None):
        super().__init__(parent)
        
        self.game_engine = game_engine
        self.dss_interface = DSSInterface(game_engine)
        
        # 游戏状态
        self.current_player = 'red'
        self.game_status = 'ready'  # ready, playing, paused, finished
        self.selected_piece = None
        self.move_history = []
        
        self.init_ui()
        self.connect_signals()
        self.setup_game()
        
        # 如果有游戏引擎，更新初始状态
        if self.game_engine:
            self.update_game_state()
    
    def update_game_state(self):
        """更新游戏状态显示"""
        if self.game_engine:
            # 从游戏引擎获取状态
            game_state = self.game_engine.get_game_state()
            
            # 更新棋盘
            self.chess_board.set_board_state(game_state['board'])
            
            # 更新当前玩家显示
            current_player = game_state['current_player']
            self.current_player_label.setText(f"当前玩家: {'红方' if current_player == 'red' else '黑方'}")
            
            # 更新游戏状态
            if game_state['game_over']:
                winner = '红方' if game_state['winner'] == 'red' else '黑方'
                self.game_status_label.setText(f"游戏结束 - {winner}获胜")
                self.status_bar.showMessage(f"游戏结束，{winner}获胜！")
            else:
                self.game_status_label.setText("游戏进行中")
        else:
            # 使用DSS接口轮询状态
            self.dss_interface.poll_game_state()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("中国象棋 - Chinese Chess")
        self.setMinimumSize(1000, 800)
        self.resize(1200, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：棋盘区域
        self.create_board_area(splitter)
        
        # 右侧：控制面板
        self.create_control_panel(splitter)
        
        # 设置分割器比例
        splitter.setSizes([700, 300])
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 设置样式
        self.set_styles()
    
    def create_board_area(self, parent):
        """创建棋盘区域"""
        board_frame = QFrame()
        board_frame.setFrameStyle(QFrame.StyledPanel)
        parent.addWidget(board_frame)
        
        board_layout = QVBoxLayout(board_frame)
        
        # 游戏标题
        title_label = QLabel("中国象棋")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("SimHei", 20, QFont.Bold))
        board_layout.addWidget(title_label)
        
        # 棋盘组件
        self.chess_board = ChessBoardWidget()
        board_layout.addWidget(self.chess_board, 1)
        
        # 当前玩家指示
        self.current_player_label = QLabel("当前玩家: 红方")
        self.current_player_label.setAlignment(Qt.AlignCenter)
        self.current_player_label.setFont(QFont("SimHei", 14))
        board_layout.addWidget(self.current_player_label)
    
    def create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        parent.addWidget(control_frame)
        
        control_layout = QVBoxLayout(control_frame)
        
        # 游戏控制按钮组
        self.create_game_controls(control_layout)
        
        # 游戏信息显示
        self.create_game_info(control_layout)
        
        # 移动历史
        self.create_move_history(control_layout)
        
        # AI状态显示
        self.create_ai_status(control_layout)
        
        control_layout.addStretch()
    
    def create_game_controls(self, parent_layout):
        """创建游戏控制按钮"""
        controls_group = QGroupBox("游戏控制")
        controls_layout = QGridLayout(controls_group)
        
        # 新游戏按钮
        self.new_game_btn = QPushButton("新游戏")
        self.new_game_btn.clicked.connect(self.new_game)
        controls_layout.addWidget(self.new_game_btn, 0, 0)
        
        # 暂停/继续按钮
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        controls_layout.addWidget(self.pause_btn, 0, 1)
        
        # 悔棋按钮
        self.undo_btn = QPushButton("悔棋")
        self.undo_btn.clicked.connect(self.undo_move)
        controls_layout.addWidget(self.undo_btn, 1, 0)
        
        # 重做按钮
        self.redo_btn = QPushButton("重做")
        self.redo_btn.clicked.connect(self.redo_move)
        controls_layout.addWidget(self.redo_btn, 1, 1)
        
        # 投降按钮
        self.surrender_btn = QPushButton("投降")
        self.surrender_btn.clicked.connect(self.surrender)
        controls_layout.addWidget(self.surrender_btn, 2, 0, 1, 2)
        
        parent_layout.addWidget(controls_group)
    
    def create_game_info(self, parent_layout):
        """创建游戏信息显示"""
        info_group = QGroupBox("游戏信息")
        info_layout = QVBoxLayout(info_group)
        
        # 游戏状态
        self.game_status_label = QLabel("状态: 准备中")
        info_layout.addWidget(self.game_status_label)
        
        # 移动计数
        self.move_count_label = QLabel("移动次数: 0")
        info_layout.addWidget(self.move_count_label)
        
        # 游戏时间
        self.game_time_label = QLabel("游戏时间: 00:00")
        info_layout.addWidget(self.game_time_label)
        
        # 计时器
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.update_game_time)
        self.game_start_time = 0
        
        parent_layout.addWidget(info_group)
    
    def create_move_history(self, parent_layout):
        """创建移动历史显示"""
        history_group = QGroupBox("移动历史")
        history_layout = QVBoxLayout(history_group)
        
        self.move_history_text = QTextEdit()
        self.move_history_text.setMaximumHeight(200)
        self.move_history_text.setReadOnly(True)
        history_layout.addWidget(self.move_history_text)
        
        parent_layout.addWidget(history_group)
    
    def create_ai_status(self, parent_layout):
        """创建AI状态显示"""
        ai_group = QGroupBox("AI状态")
        ai_layout = QVBoxLayout(ai_group)
        
        self.ai_status_label = QLabel("AI状态: 待机")
        ai_layout.addWidget(self.ai_status_label)
        
        self.ai_progress = QProgressBar()
        self.ai_progress.setVisible(False)
        ai_layout.addWidget(self.ai_progress)
        
        parent_layout.addWidget(ai_group)
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 游戏菜单
        game_menu = menubar.addMenu('游戏')
        
        new_action = QAction('新游戏', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_game)
        game_menu.addAction(new_action)
        
        game_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        game_menu.addAction(exit_action)
        
        # 设置菜单
        settings_menu = menubar.addMenu('设置')
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("准备开始游戏")
    
    def set_styles(self):
        """设置样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
    
    def connect_signals(self):
        """连接信号"""
        # 棋盘信号
        self.chess_board.piece_clicked.connect(self.on_piece_clicked)
        self.chess_board.square_clicked.connect(self.on_square_clicked)
        self.chess_board.move_requested.connect(self.on_move_requested)
        
        # DSS接口信号
        self.dss_interface.game_state_updated.connect(self.on_game_state_updated)
        self.dss_interface.move_completed.connect(self.on_move_completed)
        self.dss_interface.game_over.connect(self.on_game_over)
        self.dss_interface.error_occurred.connect(self.on_error_occurred)
        self.dss_interface.ai_thinking_status.connect(self.on_ai_thinking_status)
    
    def setup_game(self):
        """设置游戏"""
        if self.game_engine:
            self.dss_interface.set_game_engine(self.game_engine)
            self.update_board_from_engine()
    
    def new_game(self):
        """开始新游戏"""
        reply = QMessageBox.question(self, '新游戏', '确定要开始新游戏吗？',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.dss_interface.send_message(MessageType.RESET_GAME, {})
            self.reset_ui_state()
            self.start_game_timer()
    
    def reset_game(self):
        """重置游戏"""
        if self.game_engine:
            # 使用游戏引擎重置
            self.game_engine.reset_game()
            self.update_game_state()
        else:
            # 使用DSS接口重置
            self.dss_interface.send_reset_request()
        
        # 清空移动历史
        self.move_history.clear()
        self.status_bar.showMessage("游戏已重置", 2000)
    
    def toggle_pause(self):
        """切换暂停状态"""
        if self.game_status == 'playing':
            self.game_status = 'paused'
            self.pause_btn.setText("继续")
            self.game_timer.stop()
            self.status_bar.showMessage("游戏已暂停")
        elif self.game_status == 'paused':
            self.game_status = 'playing'
            self.pause_btn.setText("暂停")
            self.game_timer.start(1000)
            self.status_bar.showMessage("游戏继续")
    
    def undo_move(self):
        """悔棋"""
        # TODO: 实现悔棋功能
        pass
    
    def redo_move(self):
        """重做"""
        # TODO: 实现重做功能
        pass
    
    def surrender(self):
        """投降"""
        reply = QMessageBox.question(self, '投降', '确定要投降吗？',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.end_game(f"{self.get_opponent_color()}方获胜（对手投降）")
    
    def on_piece_clicked(self, row, col):
        """棋子被点击"""
        piece = self.chess_board.get_piece_at_position(row, col)
        if piece:
            if piece.get('color') == self.current_player:
                # 点击己方棋子，选择该棋子
                self.select_piece(row, col)
            elif self.selected_piece:
                # 点击对方棋子，如果已选择己方棋子，则尝试吃子
                self.try_move(self.selected_piece, (row, col))
    
    def on_square_clicked(self, row, col):
        """棋盘格子被点击"""
        if self.selected_piece:
            # 尝试移动到点击的位置
            self.try_move(self.selected_piece, (row, col))
    
    def on_move_requested(self, from_pos, to_pos):
        """请求移动棋子"""
        self.try_move(from_pos, to_pos)
    
    def on_piece_moved(self, from_pos, to_pos):
        """处理棋子移动"""
        if self.game_engine:
            # 使用游戏引擎处理移动
            success = self.game_engine.make_move(from_pos, to_pos)
            if success:
                self.update_game_state()
                self.add_move_to_history(from_pos, to_pos)
            else:
                self.status_bar.showMessage("无效移动", 2000)
        else:
            # 使用DSS接口发送移动请求
            self.dss_interface.send_move_request(from_pos, to_pos)
    
    def select_piece(self, row, col):
        """选择棋子"""
        self.selected_piece = (row, col)
        self.chess_board.set_selected_position((row, col))
        
        # TODO: 获取有效移动位置
        valid_moves = self.get_valid_moves(row, col)
        self.chess_board.set_valid_moves(valid_moves)
    
    def try_move(self, from_pos, to_pos):
        """尝试移动棋子"""
        if self.game_status != 'playing':
            return
        
        # 如果有游戏引擎，直接使用游戏引擎处理移动
        if self.game_engine:
            success = self.game_engine.make_move(from_pos, to_pos)
            if success:
                self.update_game_state()
                self.add_move_to_history(from_pos, to_pos)
                self.chess_board.clear_highlights()
                self.selected_piece = None
                self.status_bar.showMessage("移动成功", 1000)
            else:
                self.status_bar.showMessage("无效移动", 2000)
            return
        
        # 发送移动请求到DSS接口
        move_data = {
            'from_pos': from_pos,
            'to_pos': to_pos,
            'player': self.current_player
        }
        
        result = self.dss_interface.send_message(MessageType.MOVE_REQUEST, move_data)
        
        if result and result.get('success'):
            self.execute_move(from_pos, to_pos)
        else:
            error_msg = result.get('message', '移动失败') if result else '移动失败'
            self.status_bar.showMessage(error_msg)
    
    def execute_move(self, from_pos, to_pos):
        """执行移动"""
        # 更新棋盘显示
        self.chess_board.set_last_move(from_pos, to_pos)
        self.chess_board.clear_highlights()
        self.selected_piece = None
        
        # 切换玩家
        self.switch_player()
        
        # 添加到移动历史
        self.add_move_to_history(from_pos, to_pos)
        
        # 更新移动计数
        self.update_move_count()
    
    def switch_player(self):
        """切换当前玩家"""
        self.current_player = 'black' if self.current_player == 'red' else 'red'
        player_name = '红方' if self.current_player == 'red' else '黑方'
        self.current_player_label.setText(f"当前玩家: {player_name}")
    
    def get_opponent_color(self):
        """获取对手颜色"""
        return 'black' if self.current_player == 'red' else 'red'
    
    def get_valid_moves(self, row, col):
        """获取有效移动位置"""
        # TODO: 从游戏引擎获取有效移动
        return []
    
    def add_move_to_history(self, from_pos, to_pos):
        """添加移动到历史记录"""
        move_text = f"{len(self.move_history) + 1}. {from_pos} -> {to_pos}"
        self.move_history.append((from_pos, to_pos))
        self.move_history_text.append(move_text)
    
    def update_move_count(self):
        """更新移动计数"""
        count = len(self.move_history)
        self.move_count_label.setText(f"移动次数: {count}")
    
    def start_game_timer(self):
        """开始游戏计时"""
        import time
        self.game_start_time = time.time()
        self.game_timer.start(1000)
        self.game_status = 'playing'
    
    def update_game_time(self):
        """更新游戏时间显示"""
        import time
        elapsed = int(time.time() - self.game_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        self.game_time_label.setText(f"游戏时间: {minutes:02d}:{seconds:02d}")
    
    def reset_ui_state(self):
        """重置UI状态"""
        self.current_player = 'red'
        self.current_player_label.setText("当前玩家: 红方")
        self.selected_piece = None
        self.move_history = []
        self.move_history_text.clear()
        self.move_count_label.setText("移动次数: 0")
        self.game_time_label.setText("游戏时间: 00:00")
        self.chess_board.clear_highlights()
        self.pause_btn.setText("暂停")
        self.game_status = 'ready'
    
    def update_board_from_engine(self):
        """从游戏引擎更新棋盘"""
        if self.game_engine and hasattr(self.game_engine, 'get_board_state'):
            board_state = self.game_engine.get_board_state()
            if board_state:
                self.chess_board.set_board_state(board_state)
    
    def on_game_state_updated(self, state):
        """游戏状态更新"""
        if 'board' in state:
            self.chess_board.set_board_state(state['board'])
        
        if 'current_player' in state:
            self.current_player = state['current_player']
            player_name = '红方' if self.current_player == 'red' else '黑方'
            self.current_player_label.setText(f"当前玩家: {player_name}")
    
    def on_move_completed(self, move_data):
        """移动完成"""
        self.status_bar.showMessage("移动完成")
    
    def on_game_over(self, result):
        """游戏结束"""
        winner = result.get('winner', '未知')
        reason = result.get('reason', '游戏结束')
        self.end_game(f"{winner}获胜 - {reason}")
    
    def on_error_occurred(self, error_msg):
        """发生错误"""
        self.status_bar.showMessage(f"错误: {error_msg}")
        QMessageBox.warning(self, "错误", error_msg)
    
    def on_ai_thinking_status(self, is_thinking):
        """AI思考状态变化"""
        if is_thinking:
            self.ai_status_label.setText("AI状态: 思考中...")
            self.ai_progress.setVisible(True)
            self.ai_progress.setRange(0, 0)  # 无限进度条
        else:
            self.ai_status_label.setText("AI状态: 待机")
            self.ai_progress.setVisible(False)
    
    def end_game(self, result_text):
        """结束游戏"""
        self.game_status = 'finished'
        self.game_timer.stop()
        self.status_bar.showMessage(result_text)
        
        QMessageBox.information(self, "游戏结束", result_text)
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
                         "中国象棋游戏\n\n"
                         "基于PyQt5开发的中国象棋游戏\n"
                         "支持人机对战和AI训练\n\n"
                         "版本: 1.0.0")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.game_status == 'playing':
            reply = QMessageBox.question(self, '退出游戏', 
                                       '游戏正在进行中，确定要退出吗？',
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # 停止DSS接口轮询
        self.dss_interface.stop_polling()
        event.accept()