#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人机对抗模式
支持玩家选择红方或黑方与AI对战
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Tuple, Optional

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.game_state import GameState, Player
from board.board import Board, Position
from piece.piece import Piece, PieceType
from rule.rule import Rule, MoveResult
from ai.networks.network import ChessNet, create_chess_net
from ai.utils.encoder import BoardEncoder, ActionEncoder
from ai.algorithms.mcts import MCTS
from config.config import NetworkConfig, MCTSConfig


class AIPlayer:
    """AI玩家类"""
    
    def __init__(self, model_path: str = "ai/checkpoints/final_model.pth", device: str = "cpu"):
        """
        初始化AI玩家
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.board_encoder = BoardEncoder()
        self.action_encoder = ActionEncoder()
        
        # 加载训练好的神经网络
        self.network = create_chess_net(NetworkConfig.to_dict())
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                # 处理不同格式的检查点文件
                if 'model_state_dict' in checkpoint:
                    self.network.load_state_dict(checkpoint['model_state_dict'])
                elif 'network_state_dict' in checkpoint:
                    self.network.load_state_dict(checkpoint['network_state_dict'])
                else:
                    # 如果是直接保存的state_dict，尝试加载
                    self.network.load_state_dict(checkpoint)
                print(f"✅ 成功加载AI模型: {model_path}")
            except Exception as e:
                print(f"⚠️  模型加载失败: {e}")
                print("🔄 使用随机初始化的网络")
        else:
            print(f"⚠️  模型文件不存在: {model_path}，使用随机初始化的网络")
        
        self.network.eval()
        
        # 初始化MCTS
        self.mcts = MCTS(
            neural_network=self.network,
            board_encoder=self.board_encoder,
            action_encoder=self.action_encoder,
            c_puct=MCTSConfig.C_PUCT,
            num_simulations=MCTSConfig.GAME_NUM_SIMULATIONS,  # 对战时使用较少的模拟次数
            temperature=MCTSConfig.GAME_TEMPERATURE  # 较低的温度，更确定性的选择
        )
    
    def get_move(self, game_state: GameState) -> Tuple[Position, Position]:
        """
        获取AI的下一步移动
        
        Args:
            game_state: 当前游戏状态
            
        Returns:
            (from_pos, to_pos): 移动的起始和目标位置
        """
        print("🤖 AI正在思考中...")
        start_time = time.time()
        
        try:
            from_pos, to_pos = self.mcts.get_best_action(game_state)
            think_time = time.time() - start_time
            print(f"🤖 AI思考完成，用时 {think_time:.2f} 秒")
            return from_pos, to_pos
        except Exception as e:
            print(f"❌ AI决策出错: {e}")
            # 如果AI出错，返回一个随机的合法移动
            valid_moves = game_state.get_valid_moves()
            if valid_moves:
                move = valid_moves[0]
                return move[0], move[1]
            else:
                return None, None


class HumanVsAI:
    """人机对抗游戏类"""
    
    def __init__(self):
        """初始化游戏"""
        self.board = Board()
        self.rule = Rule(self.board)  # Rule需要board参数
        self.game_state = GameState()  # GameState不需要参数
        self.ai_player = None
        self.human_player = None
        self.ai_side = None
        
        # 设置玩家
        self.game_state.setup_players("人类玩家", "AI玩家")
        
        # 初始化棋盘
        self.rule.initialize_board()
    
    def display_board(self):
        """显示棋盘"""
        print("\n" + "="*50)
        print("   0   1   2   3   4   5   6   7   8")
        print("  " + "-"*37)
        
        for row in range(10):
            print(f"{row}|", end="")
            for col in range(9):
                pos = Position(col, row)  # 修正参数顺序：x(col)在前，y(row)在后
                piece = self.board.get_piece_at(pos)  # 使用正确的方法名
                if piece:
                    # 根据棋子类型和颜色显示不同的符号
                    piece_char = self._get_piece_display(piece)
                    print(f" {piece_char} ", end="|")
                else:
                    print("   ", end="|")
            print()
            print("  " + "-"*37)
        print("="*50)
    
    def _get_piece_display(self, piece: Piece) -> str:
        """获取棋子的显示字符"""
        piece_symbols = {
            PieceType.GENERAL: "帅" if piece.side == "red" else "将",
            PieceType.ADVISOR: "仕" if piece.side == "red" else "士",
            PieceType.ELEPHANT: "相" if piece.side == "red" else "象",
            PieceType.HORSE: "马" if piece.side == "red" else "馬",
            PieceType.CHARIOT: "车" if piece.side == "red" else "車",
            PieceType.CANNON: "炮" if piece.side == "red" else "砲",
            PieceType.SOLDIER: "兵" if piece.side == "red" else "卒",
        }
        return piece_symbols.get(piece.type, "?")
    
    def parse_position(self, pos_str: str) -> Optional[Position]:
        """解析位置字符串，支持逗号分隔格式（如'9,0'）"""
        try:
            # 支持逗号分隔格式
            if ',' in pos_str:
                parts = pos_str.split(',')
                if len(parts) != 2:
                    return None
                row = int(parts[0].strip())
                col = int(parts[1].strip())
            else:
                # 兼容原有格式（如'90'）
                if len(pos_str) != 2:
                    return None
                row = int(pos_str[0])
                col = int(pos_str[1])
            
            if 0 <= row <= 9 and 0 <= col <= 8:
                return Position(row, col)
        except ValueError:
            pass
        return None
    
    def get_human_move(self) -> Tuple[Optional[Position], Optional[Position]]:
        """获取人类玩家的移动"""
        while True:
            try:
                move_input = input("\n请输入您的移动 (格式: 起始位置 目标位置，如 '9,0 8,0'，输入 'q' 退出): ").strip()
                
                if move_input.lower() == 'q':
                    return None, None
                
                parts = move_input.split()
                if len(parts) != 2:
                    print("❌ 输入格式错误！请使用格式: 起始位置 目标位置 (如 '9,0 8,0')")
                    continue
                
                from_pos = self.parse_position(parts[0])
                to_pos = self.parse_position(parts[1])
                
                if from_pos is None or to_pos is None:
                    print("❌ 位置格式错误！请使用格式 '行,列' (行号0-9，列号0-8)")
                    continue
                
                return from_pos, to_pos
                
            except KeyboardInterrupt:
                print("\n游戏被中断")
                return None, None
            except Exception as e:
                print(f"❌ 输入错误: {e}")
                continue
    
    def choose_side(self) -> str:
        """让玩家选择执棋方"""
        while True:
            choice = input("\n请选择您要执的棋子颜色:\n1. 红方 (先手)\n2. 黑方 (后手)\n请输入 1 或 2: ").strip()
            
            if choice == '1':
                return 'red'
            elif choice == '2':
                return 'black'
            else:
                print("❌ 请输入 1 或 2")
    
    def setup_game(self):
        """设置游戏"""
        print("🎮 欢迎来到中国象棋人机对抗模式！")
        print("📋 游戏规则:")
        print("   - 位置格式: 行号,列号 (如 '9,0' 表示第9行第0列)")
        print("   - 移动格式: 起始位置 + 空格 + 目标位置，如 '9,0 8,0'")
        print("   - 输入 'q' 可以随时退出游戏")
        
        # 让玩家选择执棋方
        self.human_player = self.choose_side()
        self.ai_side = 'black' if self.human_player == 'red' else 'red'
        
        print(f"\n✅ 您选择了 {'红方 (先手)' if self.human_player == 'red' else '黑方 (后手)'}")
        print(f"🤖 AI执 {'黑方 (后手)' if self.ai_side == 'black' else '红方 (先手)'}")
        
        # 初始化AI玩家
        print("\n🔄 正在加载AI模型...")
        self.ai_player = AIPlayer()
        
        print("\n🎯 游戏开始！")
    
    def play(self):
        """开始游戏"""
        self.setup_game()
        
        while not self.game_state.is_game_over():
            self.display_board()
            
            current_player = self.game_state.current_player
            print(f"\n当前轮到: {'红方' if current_player.side == 'red' else '黑方'}")
            
            if (current_player.side == 'red' and self.human_player == 'red') or \
               (current_player.side == 'black' and self.human_player == 'black'):
                # 人类玩家回合
                print("👤 您的回合")
                from_pos, to_pos = self.get_human_move()
                
                if from_pos is None or to_pos is None:
                    print("👋 游戏结束，感谢您的参与！")
                    break
                
                # 尝试执行移动
                if self.game_state.make_move(from_pos, to_pos):
                    print(f"✅ 移动成功: {from_pos} -> {to_pos}")
                else:
                    print("❌ 无效移动，请重新输入")
                    continue
            else:
                # AI玩家回合
                print("🤖 AI的回合")
                from_pos, to_pos = self.ai_player.get_move(self.game_state)
                
                if from_pos is None or to_pos is None:
                    print("❌ AI无法找到合法移动")
                    break
                
                if self.game_state.make_move(from_pos, to_pos):
                    print(f"🤖 AI移动: {from_pos} -> {to_pos}")
                else:
                    print("❌ AI移动失败")
                    break
        
        # 游戏结束
        self.display_board()
        self._show_game_result()
    
    def _show_game_result(self):
        """显示游戏结果"""
        print("\n" + "="*50)
        print("🏁 游戏结束！")
        
        if self.game_state.winner and self.game_state.winner.side == 'red':
            winner = "红方"
            if self.human_player == 'red':
                print("🎉 恭喜您获胜！")
            else:
                print("😔 AI获胜，继续努力！")
        elif self.game_state.winner and self.game_state.winner.side == 'black':
            winner = "黑方"
            if self.human_player == 'black':
                print("🎉 恭喜您获胜！")
            else:
                print("😔 AI获胜，继续努力！")
        else:
            winner = "平局"
            print("🤝 游戏平局！")
        
        print(f"🏆 最终结果: {winner}")
        print("="*50)


def main():
    """主函数"""
    try:
        game = HumanVsAI()
        game.play()
    except KeyboardInterrupt:
        print("\n👋 游戏被中断，感谢您的参与！")
    except Exception as e:
        print(f"❌ 游戏出现错误: {e}")


if __name__ == "__main__":
    main()