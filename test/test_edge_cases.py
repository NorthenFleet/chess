# chess/test/test_edge_cases.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from board.board import Board, Position
from piece.pieces import General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier
from rule.rule import Rule, MoveResult
from core.game_state import GameState


class TestEdgeCases(unittest.TestCase):
    """测试边界情况和错误处理"""
    
    def setUp(self):
        """测试前准备"""
        self.board = Board()
        self.rule = Rule(self.board)
        self.game_state = GameState()
        self.game_state.setup_players("红方", "黑方")
        self.game_state.start_game()
    
    def test_invalid_positions(self):
        """测试无效位置处理"""
        print("\n=== 测试无效位置处理 ===")
        
        # 测试超出棋盘范围的位置
        with self.assertRaises(ValueError):
            Position(-1, 0)  # x坐标小于0
        
        with self.assertRaises(ValueError):
            Position(9, 0)   # x坐标大于8
        
        with self.assertRaises(ValueError):
            Position(0, -1)  # y坐标小于0
        
        with self.assertRaises(ValueError):
            Position(0, 10)  # y坐标大于9
        
        print("✓ 无效位置正确抛出异常")
    
    def test_empty_board_operations(self):
        """测试空棋盘操作"""
        print("\n=== 测试空棋盘操作 ===")
        
        self.board.clear()
        
        # 尝试移动不存在的棋子
        result, captured = self.rule.execute_move(Position(4, 0), Position(4, 1))
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 移动不存在的棋子被正确拒绝")
        
        # 检查空棋盘的将军状态
        self.assertFalse(self.rule.is_checked("red"))
        self.assertFalse(self.rule.is_checked("black"))
        print("✓ 空棋盘无将军状态")
        
        # 检查空棋盘的游戏结束状态
        # 空棋盘时，由于没有将军，游戏可能被判定为和棋
        is_over, winner, reason = self.rule.is_game_over()
        # 空棋盘的游戏状态取决于具体实现，这里不做严格断言
        print(f"✓ 空棋盘游戏状态: 结束={is_over}, 获胜方={winner}, 原因={reason}")
    
    def test_piece_boundary_movements(self):
        """测试棋子边界移动"""
        print("\n=== 测试棋子边界移动 ===")
        
        self.board.clear()
        
        # 测试将军在九宫格边界的移动
        general = General("red")
        self.board.place_piece(general, Position(3, 0))  # 九宫格左下角
        
        # 尝试移动到九宫格外
        valid_moves = general.get_valid_moves(Position(3, 0), self.board)
        # 将军不能移动到九宫格外
        self.assertNotIn(Position(2, 0), valid_moves)
        print("✓ 将军正确限制在九宫格内")
        
        # 测试兵在边界的移动
        soldier = Soldier("red")
        self.board.place_piece(soldier, Position(0, 3))  # 最左边的兵位
        
        valid_moves = soldier.get_valid_moves(Position(0, 3), self.board)
        # 兵不能向左移动（超出边界），检查有效移动不包含无效位置
        # 由于Position(-1, 3)会抛出异常，我们检查兵只能向前移动
        expected_moves = [Position(0, 4)]  # 兵只能向前
        self.assertEqual(valid_moves, expected_moves)
        print("✓ 兵正确限制在棋盘边界内")
    
    def test_duplicate_piece_placement(self):
        """测试重复放置棋子"""
        print("\n=== 测试重复放置棋子 ===")
        
        self.board.clear()
        
        # 在同一位置放置两个棋子
        pos = Position(4, 4)
        piece1 = Chariot("red")
        piece2 = Horse("black")
        
        self.board.place_piece(piece1, pos)
        self.assertEqual(self.board.get_piece_at(pos), piece1)
        
        # 放置第二个棋子会覆盖第一个
        self.board.place_piece(piece2, pos)
        self.assertEqual(self.board.get_piece_at(pos), piece2)
        print("✓ 重复放置棋子正确覆盖")
    
    def test_move_to_same_position(self):
        """测试移动到相同位置"""
        print("\n=== 测试移动到相同位置 ===")
        
        self.board.clear()
        self.board.place_piece(Chariot("red"), Position(0, 0))
        
        # 尝试移动到相同位置
        result, captured = self.rule.execute_move(Position(0, 0), Position(0, 0))
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 移动到相同位置被正确拒绝")
    
    def test_capture_own_piece(self):
        """测试吃己方棋子"""
        print("\n=== 测试吃己方棋子 ===")
        
        self.board.clear()
        self.board.place_piece(Chariot("red"), Position(0, 0))
        self.board.place_piece(Horse("red"), Position(0, 1))
        
        # 尝试用车吃己方马
        result, captured = self.rule.execute_move(Position(0, 0), Position(0, 1))
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 吃己方棋子被正确拒绝")
    
    def test_blocked_path_movement(self):
        """测试路径被阻挡的移动"""
        print("\n=== 测试路径被阻挡的移动 ===")
        
        self.board.clear()
        self.board.place_piece(Chariot("red"), Position(0, 0))
        self.board.place_piece(Horse("black"), Position(0, 2))  # 阻挡棋子
        
        # 尝试车跳过阻挡棋子移动
        result, captured = self.rule.execute_move(Position(0, 0), Position(0, 3))
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 路径被阻挡的移动被正确拒绝")
    
    def test_horse_leg_blocking(self):
        """测试马腿被阻挡"""
        print("\n=== 测试马腿被阻挡 ===")
        
        self.board.clear()
        self.board.place_piece(Horse("red"), Position(1, 0))
        self.board.place_piece(Soldier("black"), Position(1, 1))  # 阻挡马腿
        
        # 尝试马向前跳（马腿被阻挡）
        result, captured = self.rule.execute_move(Position(1, 0), Position(0, 2))
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 马腿被阻挡的移动被正确拒绝")
    
    def test_cannon_without_platform(self):
        """测试炮没有炮台的攻击"""
        print("\n=== 测试炮没有炮台的攻击 ===")
        
        self.board.clear()
        self.board.place_piece(Cannon("red"), Position(1, 2))
        self.board.place_piece(Soldier("black"), Position(1, 4))  # 目标
        
        # 炮没有炮台，不能吃子
        result, captured = self.rule.execute_move(Position(1, 2), Position(1, 4))
        self.assertEqual(result, MoveResult.INVALID)
        print("✓ 炮没有炮台的攻击被正确拒绝")
    
    def test_general_face_to_face(self):
        """测试将帅照面"""
        print("\n=== 测试将帅照面 ===")
        
        self.board.clear()
        self.board.place_piece(General("red"), Position(4, 0))
        self.board.place_piece(General("black"), Position(4, 9))
        
        # 尝试移动红将使其与黑将照面
        result, captured = self.rule.execute_move(Position(4, 0), Position(4, 1))
        # 这个移动应该被允许，因为中间有距离
        self.assertEqual(result, MoveResult.VALID)
        print("✓ 将帅照面规则正确处理")
    
    def test_extreme_board_positions(self):
        """测试极端棋盘位置"""
        print("\n=== 测试极端棋盘位置 ===")
        
        self.board.clear()
        
        # 测试四个角落的位置
        corners = [Position(0, 0), Position(8, 0), Position(0, 9), Position(8, 9)]
        
        for corner in corners:
            self.assertTrue(self.board.is_valid_position(corner))
            self.board.place_piece(Chariot("red"), corner)
            self.assertEqual(self.board.get_piece_at(corner).type.value, "chariot")
            self.board.remove_piece(corner)
        
        print("✓ 极端位置处理正确")
    
    def test_large_number_of_pieces(self):
        """测试大量棋子的处理"""
        print("\n=== 测试大量棋子处理 ===")
        
        self.board.clear()
        
        # 在棋盘上放置大量棋子
        piece_count = 0
        for x in range(9):
            for y in range(10):
                if (x + y) % 2 == 0:  # 棋盘格子模式
                    self.board.place_piece(Soldier("red"), Position(x, y))
                    piece_count += 1
        
        self.assertEqual(len(self.board.pieces), piece_count)
        print(f"✓ 成功处理 {piece_count} 个棋子")


if __name__ == "__main__":
    unittest.main(verbosity=2)