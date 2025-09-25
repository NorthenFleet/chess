#!/usr/bin/env python3
"""
神经网络测试脚本

加载训练好的模型并测试其性能
"""

import os
import sys
import torch
import numpy as np
from typing import List, Tuple

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ai.actor_critic_network import ActorCriticNetwork
from ai.encoder import BoardEncoder, ActionEncoder
from ai.mcts import MCTS
from board.board import Board, Position
from rule.rule import Rule
from core.game_state import GameState
from train_config import NETWORK_CONFIG, DEVICE

class NetworkTester:
    """神经网络测试器"""
    
    def __init__(self, model_path: str = "ai/checkpoints/final_model.pth"):
        """
        初始化测试器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.device = DEVICE
        
        # 初始化编码器
        self.board_encoder = BoardEncoder()
        self.action_encoder = ActionEncoder()
        
        # 初始化网络
        self.network = ActorCriticNetwork(
            input_channels=NETWORK_CONFIG['input_channels'],
            hidden_channels=NETWORK_CONFIG['hidden_channels'],
            num_residual_blocks=NETWORK_CONFIG['num_residual_blocks'],
            num_attention_heads=NETWORK_CONFIG['num_attention_heads'],
            action_space_size=NETWORK_CONFIG['action_space_size'],
            use_attention=NETWORK_CONFIG['use_attention']
        ).to(self.device)
        
        # 加载模型
        self.load_model()
        
        # 初始化游戏组件
        self.board = Board()
        self.rule = Rule(self.board)
        
    def load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            print(f"❌ 模型文件不存在: {self.model_path}")
            print("🔄 使用随机初始化的网络进行测试")
            return
            
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 成功加载模型: {self.model_path}")
                
                if 'episode' in checkpoint:
                    print(f"📊 训练轮数: {checkpoint['episode']}")
                if 'total_steps' in checkpoint:
                    print(f"🔢 总步数: {checkpoint['total_steps']}")
                    
            else:
                self.network.load_state_dict(checkpoint)
                print(f"✅ 成功加载模型参数: {self.model_path}")
                
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("🔄 使用随机初始化的网络进行测试")
    
    def test_network_inference(self):
        """测试网络推理能力"""
        print("\n🧠 测试网络推理能力...")
        
        # 创建测试用的游戏状态
        game_state = GameState()
        game_state.setup_players("测试红方", "测试黑方")
        game_state.start_game()
        
        # 编码棋盘状态
        board_encoded = self.board_encoder.encode_board(game_state.board, game_state.current_player)
        board_tensor = torch.FloatTensor(board_encoded).unsqueeze(0).to(self.device)
        
        # 网络推理
        with torch.no_grad():
            action_probs, action_logits, value = self.network(board_tensor)
            
        print(f"📊 策略输出形状: {action_logits.shape}")
        print(f"📊 价值输出形状: {value.shape}")
        print(f"📊 价值预测: {value.item():.4f}")
        print(f"📊 策略概率范围: [{action_probs.min().item():.6f}, {action_probs.max().item():.6f}]")
        
        # 分析策略分布
        top_actions = torch.topk(action_probs, k=5, dim=1)
        
        print(f"🎯 前5个最可能的动作:")
        for i in range(5):
            prob = top_actions.values[0, i]
            action_idx = top_actions.indices[0, i]
            try:
                from_pos, to_pos = self.action_encoder.decode_action(action_idx.item())
                print(f"  {i+1}. 动作 {action_idx.item()}: ({from_pos.x},{from_pos.y}) -> ({to_pos.x},{to_pos.y}), 概率: {prob.item():.4f}")
            except:
                print(f"  {i+1}. 动作 {action_idx.item()}: 解码失败, 概率: {prob.item():.4f}")
    
    def test_valid_moves_prediction(self):
        """测试合法移动预测"""
        print("\n♟️  测试合法移动预测...")
        
        # 创建测试用的游戏状态
        game_state = GameState()
        game_state.setup_players("测试红方", "测试黑方")
        game_state.start_game()
        
        # 获取所有合法移动
        valid_moves_dict = self.rule.get_all_valid_moves(game_state.current_player.side)
        
        # 转换为移动列表
        valid_moves = []
        for from_pos, to_positions in valid_moves_dict.items():
            for to_pos in to_positions:
                valid_moves.append((from_pos, to_pos))
        print(f"📋 当前合法移动数量: {len(valid_moves)}")
        
        # 编码棋盘
        board_encoded = self.board_encoder.encode_board(game_state.board, game_state.current_player)
        board_tensor = torch.FloatTensor(board_encoded).unsqueeze(0).to(self.device)
        
        # 获取网络预测
        with torch.no_grad():
            action_probs, action_logits, value = self.network(board_tensor)
            
        # 分析合法移动预测准确性
        valid_action_probs = []
        for from_pos, to_pos in valid_moves[:10]:  # 只看前10个
            try:
                action_idx = self.action_encoder.encode_action(from_pos, to_pos)
                prob = action_probs[0, action_idx].item()
                valid_action_probs.append((from_pos, to_pos, prob))
            except:
                continue
        
        # 按概率排序
        valid_action_probs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"🎯 网络对合法移动的预测 (前5个):")
        for i, (from_pos, to_pos, prob) in enumerate(valid_action_probs[:5]):
            print(f"  {i+1}. ({from_pos.x},{from_pos.y}) -> ({to_pos.x},{to_pos.y}), 概率: {prob:.4f}")
    
    def test_position_evaluation(self):
        """测试局面评估"""
        print("\n⚖️  测试局面评估...")
        
        test_positions = []
        
        # 创建初始局面
        initial_game_state = GameState()
        initial_game_state.setup_players("测试红方", "测试黑方")
        initial_game_state.start_game()
        test_positions.append(("初始局面", initial_game_state))
        
        # 可以添加更多测试局面
        # 比如优势局面、劣势局面等
        
        for name, game_state in test_positions:
            board_encoded = self.board_encoder.encode_board(game_state.board, game_state.current_player)
            board_tensor = torch.FloatTensor(board_encoded).unsqueeze(0).to(self.device)
           # 获取网络预测
            with torch.no_grad():
                action_probs, action_logits, value = self.network(board_tensor)
                
            print(f"📊 {name}: 价值评估 = {value.item():.4f}")
            
            # 价值解释
            if value.item() > 0.1:
                evaluation = "优势"
            elif value.item() < -0.1:
                evaluation = "劣势"
            else:
                evaluation = "均势"
            print(f"   解释: {game_state.current_player}方处于{evaluation}")
    
    def test_network_performance(self):
        """测试网络性能"""
        print("\n⚡ 测试网络性能...")
        
        # 创建测试用的游戏状态
        game_state = GameState()
        game_state.setup_players("测试红方", "测试黑方")
        game_state.start_game()
        board_encoded = self.board_encoder.encode_board(game_state.board, game_state.current_player)
        board_tensor = torch.FloatTensor(board_encoded).unsqueeze(0).to(self.device)
        
        # 性能测试
        import time
        
        num_inferences = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_inferences):
                action_probs, action_logits, value = self.network(board_tensor)
                
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_inferences * 1000  # 毫秒
        print(f"🚀 平均推理时间: {avg_time:.2f} ms")
        print(f"🔢 每秒推理次数: {1000/avg_time:.1f} inferences/sec")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("🧪 神经网络测试开始")
        print("=" * 60)
        print(f"🖥️  设备: {self.device}")
        print(f"📁 模型路径: {self.model_path}")
        print(f"🧠 网络参数: {sum(p.numel() for p in self.network.parameters()):,}")
        
        try:
            self.test_network_inference()
            self.test_valid_moves_prediction()
            self.test_position_evaluation()
            self.test_network_performance()
            
            print("\n" + "=" * 60)
            print("✅ 所有测试完成！")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试神经网络')
    parser.add_argument('--model', type=str, default='ai/checkpoints/final_model.pth',
                       help='模型文件路径')
    
    args = parser.parse_args()
    
    # 创建测试器并运行测试
    tester = NetworkTester(args.model)
    tester.run_all_tests()

if __name__ == "__main__":
    main()