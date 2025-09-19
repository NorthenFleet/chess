#!/usr/bin/env python3
"""
中国象棋AI演示脚本

这个脚本提供了一个简单的命令行界面来演示AI功能。
支持以下功能：
1. 获取AI推荐的最佳移动
2. 评估当前局面
3. 与AI对弈

使用方法:
    python ai/demo.py --help
    python ai/demo.py --model path/to/model.pth
"""

import os
import sys
import argparse
import random
from typing import Optional, Tuple

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入基础模块
from core.game_state import GameState
from board.board import Board, Position

# 尝试导入PyTorch和numpy
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: PyTorch未安装，AI功能将受限")
    TORCH_AVAILABLE = False

# 尝试导入AI模块
try:
    from ai.network import ChessNet
    from ai.encoder import BoardEncoder, ActionEncoder
    from ai.mcts import MCTS
    AI_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: AI模块导入失败，将使用简化版本")
    AI_MODULES_AVAILABLE = False


class ChessAI:
    """中国象棋AI类"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        初始化AI
        
        Args:
            model_path: 模型文件路径，如果为None则使用随机初始化模型
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.simplified_mode = not (TORCH_AVAILABLE and AI_MODULES_AVAILABLE)
        
        # 初始化棋盘和规则（AI需要这些来分析局面）
        from board.board import Board
        from rule.rule import Rule
        self.board = Board()
        self.rule = Rule(self.board)
        
        if self.simplified_mode:
            print("警告: PyTorch或AI模块未完全可用，将使用简化版本")
            return
            
        self.board_encoder = BoardEncoder()
        self.action_encoder = ActionEncoder()
        
        # 初始化神经网络
        self.network = ChessNet()
        
        if model_path and os.path.exists(model_path):
            try:
                self.network.load_state_dict(torch.load(model_path, map_location=device))
                print(f"成功加载模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}，使用随机初始化模型")
        else:
            print("使用随机初始化模型")
        
        self.network.to(device)
        self.network.eval()
        
        # 初始化MCTS
        self.mcts = MCTS(self.network, self.board_encoder, self.action_encoder)
    
    def get_best_move(self, game_state: GameState, simulations: int = 800) -> Optional[Tuple[Position, Position]]:
        """
        获取最佳移动
        
        Args:
            game_state: 游戏状态
            simulations: MCTS模拟次数
            
        Returns:
            最佳移动 (起始位置, 目标位置)，如果无合法移动则返回None
        """
        if self.simplified_mode:
            # 简化模式：返回随机合法移动
            return self._get_random_move(game_state)
            
        try:
            # 使用MCTS搜索最佳移动
            action_probs = self.mcts.search(game_state, simulations)
            
            if not action_probs:
                return None
            
            # 选择概率最高的动作
            best_action = max(action_probs.items(), key=lambda x: x[1])[0]
            return self.action_encoder.decode_action(best_action)
            
        except Exception as e:
            print(f"MCTS搜索失败: {e}，使用随机移动")
            return self._get_random_move(game_state)
    
    def _get_random_move(self, game_state: GameState) -> Optional[Tuple[Position, Position]]:
        """获取随机合法移动"""
        # 需要先同步棋盘状态到AI的board实例
        # 这里假设game_state有board属性，如果没有需要其他方式获取棋盘状态
        if hasattr(game_state, 'board'):
            # 复制棋盘状态
            self.board.pieces = game_state.board.pieces.copy()
            self.board.piece_positions = game_state.board.piece_positions.copy()
        else:
            # 如果没有board属性，使用默认初始化
            from rule.rule import Rule
            rule = Rule(self.board)
            rule.initialize_board()
        
        # 使用Rule类获取合法移动
        current_player_side = game_state.current_player.side
        valid_moves = self.rule.get_all_valid_moves(current_player_side)
        
        if not valid_moves:
            return None
            
        # 将字典转换为移动列表
        all_moves = []
        for from_pos, to_positions in valid_moves.items():
            for to_pos in to_positions:
                all_moves.append((from_pos, to_pos))
        
        if not all_moves:
            return None
            
        return random.choice(all_moves)
    
    def evaluate_position(self, game_state: GameState) -> float:
        """
        评估当前局面
        
        Args:
            game_state: 游戏状态
            
        Returns:
            局面评估值 (-1到1之间，正值表示当前玩家优势)
        """
        if self.simplified_mode:
            # 简化模式：返回随机评估值
            return random.uniform(-0.5, 0.5)
            
        if not TORCH_AVAILABLE:
            return 0.0  # 无法评估时返回中性值
            
        try:
            # 编码棋盘状态
            board_tensor = self.board_encoder.encode_board(
                game_state.board, 
                game_state.current_player
            )
            
            # 转换为PyTorch张量
            input_tensor = torch.FloatTensor(board_tensor).unsqueeze(0).to(self.device)
            
            # 获取网络预测
            with torch.no_grad():
                _, value = self.network(input_tensor)
                return float(value.item())
                
        except Exception as e:
            print(f"局面评估失败: {e}")
            return 0.0


def play_human_vs_ai(ai: ChessAI, human_color: str = "red"):
    """人机对弈"""
    # 创建完整的游戏实例
    from board.board import Board
    from rule.rule import Rule
    
    board = Board()
    rule = Rule(board)
    game_state = GameState()
    
    # 设置玩家
    if human_color == "red":
        game_state.setup_players("人类玩家", "AI")
    else:
        game_state.setup_players("AI", "人类玩家")
    
    # 初始化棋盘
    rule.initialize_board()
    
    # 开始游戏
    game_state.start_game()
    
    print("\n=== 中国象棋人机对弈 ===")
    print(f"人类执: {'红' if human_color == 'red' else '黑'}方")
    print("输入格式: 起始位置 目标位置 (例如: 0,0 0,1)")
    print("输入 'quit' 退出游戏\n")
    
    while not game_state.game_over:
        # 显示棋盘
        print(board)
        print(f"当前玩家: {game_state.current_player.name} ({game_state.current_player.side})")
        
        # 创建临时GameState给AI使用，包含board信息
        temp_game_state = GameState()
        temp_game_state.current_player = game_state.current_player
        temp_game_state.board = board  # 添加board属性
        
        # 评估当前局面
        position_value = ai.evaluate_position(temp_game_state)
        print(f"AI评估: {position_value:.3f} (正值对红方有利)")
        
        if game_state.current_player.side == human_color:
            # 人类回合
            print("\n请输入您的移动:")
            
            while True:
                try:
                    user_input = input("> ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("游戏结束")
                        return
                    
                    # 解析输入
                    parts = user_input.split()
                    if len(parts) != 2:
                        print("输入格式错误，请使用: 起始位置 目标位置")
                        continue
                    
                    from_str, to_str = parts
                    from_row, from_col = map(int, from_str.split(','))
                    to_row, to_col = map(int, to_str.split(','))
                    
                    from_pos = Position(from_row, from_col)
                    to_pos = Position(to_row, to_col)
                    
                    # 验证并执行移动
                    result, captured = rule.execute_move(from_pos, to_pos)
                    if result.value == "valid" or result.value == "capture":
                        print(f"移动成功: {from_pos} -> {to_pos}")
                        if captured:
                            print(f"吃掉了对方的 {captured.name}")
                        
                        # 检查游戏是否结束
                        is_over, winner_side, reason = rule.is_game_over()
                        if is_over:
                            if winner_side:
                                winner = game_state.players[winner_side]
                                game_state.end_game(winner, reason)
                            else:
                                game_state.end_game(None, reason)
                        else:
                            game_state.switch_turn()
                        break
                    else:
                        print(f"无效移动: {result.value}")
                        
                except (ValueError, IndexError):
                    print("输入格式错误，请使用: 起始位置 目标位置 (例如: 0,0 0,1)")
                except KeyboardInterrupt:
                    print("\n游戏结束")
                    return
                except Exception as e:
                    print(f"移动失败: {e}")
        else:
            # AI回合
            print("\nAI正在思考...")
            
            try:
                best_move = ai.get_best_move(temp_game_state)
                if best_move:
                    from_pos, to_pos = best_move
                    result, captured = rule.execute_move(from_pos, to_pos)
                    if result.value == "valid" or result.value == "capture":
                        print(f"AI移动: {from_pos} -> {to_pos}")
                        if captured:
                            print(f"AI吃掉了您的 {captured.name}")
                        
                        # 检查游戏是否结束
                        is_over, winner_side, reason = rule.is_game_over()
                        if is_over:
                            if winner_side:
                                winner = game_state.players[winner_side]
                                game_state.end_game(winner, reason)
                            else:
                                game_state.end_game(None, reason)
                        else:
                            game_state.switch_turn()
                    else:
                        print(f"AI移动失败: {result.value}")
                        break
                else:
                    print("AI无法找到合法移动")
                    break
                    
            except Exception as e:
                print(f"AI出现错误: {e}")
                break
        
        print("\n" + "="*50)
    
    # 游戏结束
    print("\n游戏结束！")
    if game_state.winner:
        if game_state.winner.side == human_color:
            print("恭喜！您获胜了！")
        else:
            print("AI获胜！")
    else:
        print("平局！")


def ai_vs_ai(ai1: ChessAI, ai2: ChessAI, num_games: int = 10):
    """AI对战"""
    print(f"\n=== AI对战 ({num_games} 局) ===")
    
    ai1_wins = 0
    ai2_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        print(f"\n第 {game_idx + 1} 局:")
        
        # 创建完整的游戏实例
        from board.board import Board
        from rule.rule import Rule
        
        board = Board()
        rule = Rule(board)
        game_state = GameState()
        
        # 设置玩家
        game_state.setup_players("AI1", "AI2")
        
        # 初始化棋盘
        rule.initialize_board()
        
        # 开始游戏
        game_state.start_game()
        
        move_count = 0
        max_moves = 200
        
        while not game_state.game_over and move_count < max_moves:
            current_ai = ai1 if game_state.current_player.side == "red" else ai2
            
            try:
                # 创建临时GameState给AI使用，包含board信息
                temp_game_state = GameState()
                temp_game_state.current_player = game_state.current_player
                temp_game_state.board = board  # 添加board属性
                
                best_move = current_ai.get_best_move(temp_game_state)
                if best_move:
                    from_pos, to_pos = best_move
                    
                    # 验证并执行移动
                    result, captured = rule.execute_move(from_pos, to_pos)
                    if result.value == "valid" or result.value == "capture":
                        move_count += 1
                        
                        # 检查游戏是否结束
                        is_over, winner_side, reason = rule.is_game_over()
                        if is_over:
                            if winner_side:
                                winner = game_state.players[winner_side]
                                game_state.end_game(winner, reason)
                            else:
                                game_state.end_game(None, reason)
                        else:
                            game_state.switch_turn()
                    else:
                        print(f"AI出现非法移动: {from_pos} -> {to_pos} ({result.value})")
                        break
                else:
                    print("AI无法找到合法移动")
                    break
                
            except Exception as e:
                print(f"AI出现错误: {e}")
                break
        
        # 统计结果
        if game_state.winner is None:
            draws += 1
            print("平局")
        elif game_state.winner.side == "red":
            ai1_wins += 1
            print("AI1 (红方) 获胜")
        else:
            ai2_wins += 1
            print("AI2 (黑方) 获胜")
    
    # 打印统计结果
    print(f"\n=== 对战结果 ===")
    print(f"AI1 获胜: {ai1_wins} 局 ({ai1_wins/num_games*100:.1f}%)")
    print(f"AI2 获胜: {ai2_wins} 局 ({ai2_wins/num_games*100:.1f}%)")
    print(f"平局: {draws} 局 ({draws/num_games*100:.1f}%)")


def train_ai(iterations: int = 10, checkpoint_dir: str = "checkpoints"):
    """训练AI"""
    if not TORCH_AVAILABLE or not AI_MODULES_AVAILABLE:
        print("错误: 训练模式需要完整的AI模块和PyTorch")
        return
        
    print(f"\n=== 开始训练AI ({iterations} 次迭代) ===")
    
    # 检查是否有GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建训练器
    network_config = {
        'hidden_channels': 128,  # 减少通道数以加快训练
        'num_residual_blocks': 10,
        'use_attention': True
    }
    
    training_config = {
        'mcts_simulations': 200,  # 减少模拟次数
        'self_play_games': 20,
        'epochs_per_iteration': 5,
        'batch_size': 16
    }
    
    try:
        from ai.trainer import ChessTrainer
        trainer = ChessTrainer(
            network_config=network_config,
            training_config=training_config,
            device=device
        )
        
        # 开始训练
        trainer.train(iterations, checkpoint_dir)
        
        print(f"训练完成！模型已保存到 {checkpoint_dir}")
    except ImportError:
        print("错误: 无法导入训练模块")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中国象棋AI演示')
    parser.add_argument('--mode', choices=['train', 'play', 'ai_vs_ai'], 
                       default='play', help='运行模式')
    parser.add_argument('--model', type=str, help='预训练模型路径')
    parser.add_argument('--iterations', type=int, default=10, help='训练迭代次数')
    parser.add_argument('--games', type=int, default=10, help='AI对战局数')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='计算设备')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练模式
        train_ai(args.iterations)
        
    elif args.mode == 'play':
        # 人机对弈模式
        ai = ChessAI(args.model, args.device)
        play_human_vs_ai(ai)
        
    elif args.mode == 'ai_vs_ai':
        # AI对战模式
        ai1 = ChessAI(args.model, args.device)
        ai2 = ChessAI(args.model, args.device)
        ai_vs_ai(ai1, ai2, args.games)


if __name__ == "__main__":
    main()