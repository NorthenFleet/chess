# chess/test/test_endgame.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from board.board import Board, Position
from piece.pieces import General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier
from rule.rule import Rule, MoveResult
from core.game_state import GameState


class TestEndgameLogic(unittest.TestCase):
    """测试胜负判断逻辑"""
    
    def setUp(self):
        """测试前准备"""
        self.board = Board()
        self.rule = Rule(self.board)
        self.game_state = GameState()
        self.game_state.setup_players("红方", "黑方")
        self.game_state.start_game()
    
    def test_simple_checkmate(self):
        """测试简单将死局面"""
        print("\n=== 测试简单将死 ===")
        
        self.board.clear()
        
        # 设置一个真正的将死局面
        # 黑方将军被困在角落
        self.board.place_piece(General("black"), Position(4, 9))
        self.board.place_piece(General("red"), Position(4, 7))  # 红方将军
        
        # 用三个车完全封锁黑方将军
        self.board.place_piece(Chariot("red"), Position(3, 9))  # 左侧
        self.board.place_piece(Chariot("red"), Position(5, 9))  # 右侧
        self.board.place_piece(Chariot("red"), Position(4, 8))  # 正面将军
        # 再加一个车封锁后路
        self.board.place_piece(Chariot("red"), Position(3, 8))  # 封锁左后方
        self.board.place_piece(Chariot("red"), Position(5, 8))  # 封锁右后方
        
        # 验证黑方被将军
        self.assertTrue(self.rule.is_checked("black"))
        print("✓ 黑方被将军")
        
        # 验证黑方被将死
        self.assertTrue(self.rule.is_checkmate("black"))
        print("✓ 黑方被将死")
        
        # 验证游戏结束
        is_over, winner, reason = self.rule.is_game_over()
        self.assertTrue(is_over)
        self.assertEqual(winner, "red")
        print(f"✓ 游戏结束，获胜方: {winner}, 原因: {reason}")
    
    def test_check_but_not_checkmate(self):
        """测试将军但不是将死的情况"""
        print("\n=== 测试将军但非将死 ===")
        
        self.board.clear()
        
        # 设置一个将军但不是将死的局面
        self.board.place_piece(General("black"), Position(4, 9))
        self.board.place_piece(General("red"), Position(4, 7))
        self.board.place_piece(Chariot("red"), Position(4, 8))  # 将军
        
        # 验证黑方被将军
        self.assertTrue(self.rule.is_checked("black"))
        print("✓ 黑方被将军")
        
        # 验证黑方没有被将死（可以左右移动）
        self.assertFalse(self.rule.is_checkmate("black"))
        print("✓ 黑方未被将死（可以左右移动）")
        
        # 验证游戏未结束
        is_over, winner, reason = self.rule.is_game_over()
        self.assertFalse(is_over)
        print("✓ 游戏继续进行")
    
    def test_no_check_situation(self):
        """测试无将军情况"""
        print("\n=== 测试无将军情况 ===")
        
        self.board.clear()
        
        # 设置一个正常对局局面
        self.board.place_piece(General("red"), Position(4, 0))
        self.board.place_piece(General("black"), Position(4, 9))
        self.board.place_piece(Chariot("red"), Position(0, 0))
        self.board.place_piece(Chariot("black"), Position(8, 9))
        
        # 验证双方都没有被将军
        self.assertFalse(self.rule.is_checked("red"))
        self.assertFalse(self.rule.is_checked("black"))
        print("✓ 双方都没有被将军")
        
        # 验证游戏未结束
        is_over, winner, reason = self.rule.is_game_over()
        self.assertFalse(is_over)
        print("✓ 游戏正常进行")
    
    def test_move_into_check_prevention(self):
        """测试防止移动后被将军"""
        print("\n=== 测试防止移动后被将军 ===")
        
        self.board.clear()
        
        # 设置一个局面，如果将军移动会被对方将军
        self.board.place_piece(General("red"), Position(4, 0))
        self.board.place_piece(General("black"), Position(4, 9))
        self.board.place_piece(Chariot("black"), Position(4, 1))  # 黑车威胁红将
        
        # 红将已经被将军，尝试移动到安全位置应该是有效的
        result, _ = self.rule.execute_move(Position(4, 0), Position(3, 0))
        self.assertEqual(result, MoveResult.VALID)
        print("✓ 正确允许了脱离将军的移动")
    
    def test_checkmate_with_multiple_pieces(self):
        """测试多子配合的将死"""
        print("\n=== 测试多子配合将死 ===")
        
        self.board.clear()
        
        # 设置一个复杂的将死局面
        self.board.place_piece(General("black"), Position(4, 9))
        self.board.place_piece(General("red"), Position(4, 7))
        
        # 使用多种棋子配合将死
        self.board.place_piece(Chariot("red"), Position(4, 8))  # 车将军
        self.board.place_piece(Chariot("red"), Position(3, 9))  # 封锁左侧
        self.board.place_piece(Chariot("red"), Position(5, 9))  # 封锁右侧
        self.board.place_piece(Chariot("red"), Position(3, 8))  # 封锁左后方
        self.board.place_piece(Chariot("red"), Position(5, 8))  # 封锁右后方
        
        # 验证将死
        self.assertTrue(self.rule.is_checked("black"))
        self.assertTrue(self.rule.is_checkmate("black"))
        print("✓ 多子配合将死成功")
    
    def test_stalemate_detection(self):
        """测试和棋检测"""
        print("\n=== 测试和棋检测 ===")
        
        self.board.clear()
        
        # 设置一个真正的和棋局面（黑方无子可动但不被将军）
        self.board.place_piece(General("red"), Position(4, 0))
        self.board.place_piece(General("black"), Position(4, 9))  # 黑将在底线
        
        # 用车封锁黑方将军的所有移动路线，但不将军
        self.board.place_piece(Chariot("red"), Position(3, 8))  # 封锁左侧
        self.board.place_piece(Chariot("red"), Position(5, 8))  # 封锁右侧
        # 不放置正面的车，避免将军
        
        # 在这种情况下，黑方无法移动但不被将军
        self.assertFalse(self.rule.is_checked("black"))
        self.assertTrue(self.rule.is_stalemate("black"))
        print("✓ 正确识别和棋局面")
    
    def test_game_over_scenarios(self):
        """测试各种游戏结束场景"""
        print("\n=== 测试游戏结束场景 ===")
        
        # 场景1：将死
        self.board.clear()
        self.board.place_piece(General("red"), Position(4, 0))
        self.board.place_piece(General("black"), Position(4, 9))
        # 创建真正的将死局面
        self.board.place_piece(Chariot("red"), Position(4, 8))  # 将军
        self.board.place_piece(Chariot("red"), Position(3, 9))  # 封锁左侧
        self.board.place_piece(Chariot("red"), Position(5, 9))  # 封锁右侧
        self.board.place_piece(Chariot("red"), Position(3, 8))  # 封锁左后方
        self.board.place_piece(Chariot("red"), Position(5, 8))  # 封锁右后方
        
        is_over, winner, reason = self.rule.is_game_over()
        self.assertTrue(is_over)
        self.assertEqual(winner, "red")
        self.assertIn("将死", reason)
        print(f"✓ 将死场景: {reason}")
        
        # 场景2：正常对局
        self.board.clear()
        self.board.place_piece(General("red"), Position(4, 0))
        self.board.place_piece(General("black"), Position(4, 9))
        
        is_over, winner, reason = self.rule.is_game_over()
        self.assertFalse(is_over)
        print("✓ 正常对局继续")


if __name__ == "__main__":
    unittest.main(verbosity=2)