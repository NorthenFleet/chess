# chess/test/test_detailed.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from board.board import Board, Position
from piece.piece import PieceType
from piece.pieces import General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier
from rule.rule import Rule, MoveResult
from core.game_state import GameState, Player


class TestChineseChessDetailed(unittest.TestCase):
    """详细的中国象棋测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.board = Board()
        self.rule = Rule(self.board)
        self.game_state = GameState()
        
        # 设置玩家
        self.game_state.setup_players("红方", "黑方")
        self.game_state.start_game()
    
    def test_piece_movement_rules(self):
        """测试各种棋子的移动规则"""
        print("\n=== 测试棋子移动规则 ===")
        
        # 清空棋盘
        self.board.clear()
        
        # 测试车的移动
        print("测试车的移动...")
        chariot = Chariot("red")
        self.board.place_piece(chariot, Position(0, 0))
        
        # 车应该能直线移动
        valid_moves = chariot.get_valid_moves(Position(0, 0), self.board)
        self.assertIn(Position(0, 5), valid_moves)  # 垂直移动
        self.assertIn(Position(5, 0), valid_moves)  # 水平移动
        self.assertNotIn(Position(1, 1), valid_moves)  # 不能斜移
        print("✓ 车的移动规则正确")
        
        # 测试马的移动
        print("测试马的移动...")
        self.board.clear()
        horse = Horse("red")
        self.board.place_piece(horse, Position(1, 0))
        
        valid_moves = horse.get_valid_moves(Position(1, 0), self.board)
        self.assertIn(Position(0, 2), valid_moves)  # 日字形移动
        self.assertIn(Position(2, 2), valid_moves)  # 日字形移动
        print("✓ 马的移动规则正确")
        
        # 测试将的移动
        print("测试将的移动...")
        self.board.clear()
        general = General("red")
        self.board.place_piece(general, Position(4, 0))
        
        valid_moves = general.get_valid_moves(Position(4, 0), self.board)
        self.assertIn(Position(4, 1), valid_moves)  # 向上移动
        self.assertIn(Position(3, 0), valid_moves)  # 向左移动
        self.assertIn(Position(5, 0), valid_moves)  # 向右移动
        # 将不能移出九宫格
        self.assertNotIn(Position(2, 0), valid_moves)
        print("✓ 将的移动规则正确")
    
    def test_capture_rules(self):
        """测试吃子规则"""
        print("\n=== 测试吃子规则 ===")
        
        self.board.clear()
        
        # 放置红方车和黑方兵
        red_chariot = Chariot("red")
        black_soldier = Soldier("black")
        
        self.board.place_piece(red_chariot, Position(0, 0))
        self.board.place_piece(black_soldier, Position(0, 3))
        
        # 测试车吃兵
        result, captured = self.rule.execute_move(Position(0, 0), Position(0, 3))
        self.assertEqual(result, MoveResult.CAPTURE)
        self.assertEqual(captured.type, PieceType.SOLDIER)
        self.assertEqual(captured.side, "black")
        
        # 验证车移动到了目标位置，兵被吃掉了
        moved_piece = self.board.get_piece_at(Position(0, 3))
        self.assertIsNotNone(moved_piece)
        self.assertEqual(moved_piece.type, PieceType.CHARIOT)
        self.assertEqual(moved_piece.side, "red")
        
        # 验证原位置为空
        self.assertIsNone(self.board.get_piece_at(Position(0, 0)))
        print("✓ 吃子规则正确")
    
    def test_check_detection(self):
        """测试将军检测"""
        print("\n=== 测试将军检测 ===")
        
        self.board.clear()
        
        # 设置一个将军局面
        self.board.place_piece(General("red"), Position(4, 0))
        self.board.place_piece(General("black"), Position(4, 9))
        self.board.place_piece(Chariot("red"), Position(4, 8))
        
        # 红方车将军黑方
        self.assertTrue(self.rule.is_checked("black"))
        self.assertFalse(self.rule.is_checked("red"))
        print("✓ 将军检测正确")
    
    def test_checkmate_detection(self):
        """测试将死检测"""
        print("\n=== 测试将死检测 ===")
        
        self.board.clear()
        
        # 设置一个将死局面
        self.board.place_piece(General("red"), Position(4, 0))
        self.board.place_piece(General("black"), Position(4, 9))
        
        # 用多个车围困黑方将军
        self.board.place_piece(Chariot("red"), Position(4, 8))  # 正面将军
        self.board.place_piece(Chariot("red"), Position(3, 9))  # 左侧封锁
        self.board.place_piece(Chariot("red"), Position(5, 9))  # 右侧封锁
        self.board.place_piece(Chariot("red"), Position(4, 7))  # 后方封锁
        
        # 检测将死
        self.assertTrue(self.rule.is_checkmate("black"))
        print("✓ 将死检测正确")
    
    def test_invalid_moves(self):
        """测试无效移动"""
        print("\n=== 测试无效移动 ===")
        
        self.rule.initialize_board()
        
        # 测试移动不存在的棋子
        result, _ = self.rule.execute_move(Position(4, 4), Position(4, 5))
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 移动空位置被正确拒绝")
        
        # 测试不符合棋子规则的移动
        result, _ = self.rule.execute_move(Position(0, 0), Position(1, 1))  # 车不能斜移
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 不符合棋子规则的移动被正确拒绝")
        
        # 测试吃己方棋子
        result, _ = self.rule.execute_move(Position(0, 0), Position(1, 0))  # 红车吃红马
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 吃己方棋子被正确拒绝")
    
    def test_game_flow(self):
        """测试游戏流程"""
        print("\n=== 测试游戏流程 ===")
        
        self.rule.initialize_board()
        
        # 测试红方先行
        self.assertEqual(self.game_state.current_player.side, "red")
        print("✓ 红方先行正确")
        
        # 执行一步有效移动 - 红兵向前移动
        result, _ = self.rule.execute_move(Position(0, 3), Position(0, 4))  # 红兵前进
        self.assertEqual(result, MoveResult.VALID)
        
        # 切换回合
        self.game_state.switch_turn()
        self.assertEqual(self.game_state.current_player.side, "black")
        print("✓ 回合切换正确")
        
        # 黑方移动
        result, _ = self.rule.execute_move(Position(0, 6), Position(0, 5))  # 黑卒前进
        self.assertEqual(result, MoveResult.VALID)
        print("✓ 游戏流程正确")


if __name__ == "__main__":
    unittest.main(verbosity=2)