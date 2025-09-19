"""
蒙特卡洛树搜索(MCTS)实现

用于中国象棋的智能搜索算法
"""

import math
import numpy as np
import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.game_state import GameState
from board.board import Position
from .encoder import BoardEncoder, ActionEncoder


@dataclass
class MCTSNode:
    """MCTS树节点"""
    
    # 游戏状态
    game_state: GameState
    
    # 父节点
    parent: Optional['MCTSNode'] = None
    
    # 子节点字典 {action_idx: MCTSNode}
    children: Dict[int, 'MCTSNode'] = None
    
    # 访问次数
    visit_count: int = 0
    
    # 价值总和
    value_sum: float = 0.0
    
    # 先验概率 (来自神经网络策略输出)
    prior_prob: float = 0.0
    
    # 是否已展开
    is_expanded: bool = False
    
    # 合法动作列表
    valid_actions: List[int] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.children is None:
            self.children = {}
        if self.valid_actions is None:
            self.valid_actions = []
    
    @property
    def q_value(self) -> float:
        """平均价值 (Q值)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return len(self.children) == 0
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[int, 'MCTSNode']:
        """
        使用UCB公式选择子节点
        
        Args:
            c_puct: 探索常数
            
        Returns:
            (动作索引, 子节点)
        """
        best_action = None
        best_value = -float('inf')
        
        for action_idx, child in self.children.items():
            # UCB公式: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            exploration_term = c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
            ucb_value = child.q_value + exploration_term
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action_idx
        
        return best_action, self.children[best_action]
    
    def expand(self, action_probs: np.ndarray, valid_actions: List[int]):
        """
        展开节点，添加子节点
        
        Args:
            action_probs: 动作概率分布
            valid_actions: 合法动作列表
        """
        self.valid_actions = valid_actions
        self.is_expanded = True
        
        # 为每个合法动作创建子节点
        for action_idx in valid_actions:
            if action_idx < len(action_probs):
                prior_prob = action_probs[action_idx]
            else:
                prior_prob = 0.0
            
            # 创建子节点（暂时不创建游戏状态，延迟到需要时）
            child = MCTSNode(
                game_state=None,  # 延迟创建
                parent=self,
                prior_prob=prior_prob
            )
            self.children[action_idx] = child
    
    def backup(self, value: float):
        """
        回传价值到根节点
        
        Args:
            value: 叶子节点的价值评估
        """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # 对手视角，价值取反
            self.parent.backup(-value)
    
    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        根据访问次数计算动作概率分布
        
        Args:
            temperature: 温度参数，控制探索程度
            
        Returns:
            动作概率分布
        """
        action_probs = np.zeros(8100)  # 动作空间大小
        
        # 检查是否有子节点
        if not self.children:
            # 如果没有子节点，返回均匀分布在有效动作上
            if self.valid_actions:
                prob = 1.0 / len(self.valid_actions)
                for action in self.valid_actions:
                    action_probs[action] = prob
            return action_probs
        
        if temperature == 0:
            # 贪婪选择
            best_action = max(self.children.keys(), 
                            key=lambda a: self.children[a].visit_count)
            action_probs[best_action] = 1.0
        else:
            # 根据访问次数的温度缩放
            visits = np.array([self.children[action].visit_count 
                             for action in self.children.keys()])
            
            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            
            # 归一化
            if visits.sum() > 0:
                visits = visits / visits.sum()
                for i, action in enumerate(self.children.keys()):
                    action_probs[action] = visits[i]
        
        return action_probs


class MCTS:
    """蒙特卡洛树搜索"""
    
    def __init__(self, 
                 neural_network,
                 board_encoder: BoardEncoder,
                 action_encoder: ActionEncoder,
                 c_puct: float = 1.0,
                 num_simulations: int = 800,
                 temperature: float = 1.0):
        """
        初始化MCTS
        
        Args:
            neural_network: 神经网络模型
            board_encoder: 棋盘编码器
            action_encoder: 动作编码器
            c_puct: UCB探索常数
            num_simulations: 模拟次数
            temperature: 温度参数
        """
        self.neural_network = neural_network
        self.board_encoder = board_encoder
        self.action_encoder = action_encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        
        # 根节点
        self.root: Optional[MCTSNode] = None
    
    def search(self, game_state: GameState) -> Tuple[np.ndarray, float]:
        """
        执行MCTS搜索
        
        Args:
            game_state: 当前游戏状态
            
        Returns:
            (动作概率分布, 根节点价值)
        """
        # 创建根节点
        self.root = MCTSNode(game_state=game_state.copy())
        
        # 执行模拟
        for _ in range(self.num_simulations):
            self._simulate()
        
        # 获取动作概率分布
        action_probs = self.root.get_action_probs(self.temperature)
        root_value = self.root.q_value
        
        return action_probs, root_value
    
    def _simulate(self):
        """执行一次模拟"""
        node = self.root
        path = [node]
        
        # 1. 选择阶段：从根节点向下选择到叶子节点
        while not node.is_leaf and node.is_expanded:
            action_idx, node = node.select_child(self.c_puct)
            
            # 延迟创建游戏状态
            if node.game_state is None:
                parent_state = path[-1].game_state
                from_pos, to_pos = self.action_encoder.decode_action(action_idx)
                
                # 复制父状态并执行动作
                node.game_state = parent_state.copy()
                success = node.game_state.make_move(from_pos, to_pos)
                
                if not success:
                    # 非法移动，给予负价值
                    self._backup_path(path, -1.0)
                    return
            
            path.append(node)
        
        # 2. 展开和评估阶段
        game_state = node.game_state
        
        # 检查游戏是否结束
        if game_state.game_over:
            # 游戏结束，使用真实结果
            winner = game_state.get_winner()
            if winner is None:
                value = 0.0  # 平局
            elif winner == game_state.current_player:
                value = 1.0  # 当前玩家获胜
            else:
                value = -1.0  # 当前玩家失败
        else:
            # 使用神经网络评估
            value = self._evaluate_node(node)
        
        # 3. 回传阶段
        self._backup_path(path, value)
    
    def _evaluate_node(self, node: MCTSNode) -> float:
        """
        使用神经网络评估节点
        
        Args:
            node: 要评估的节点
            
        Returns:
            节点价值
        """
        game_state = node.game_state
        
        # 编码棋盘状态
        board_encoded = self.board_encoder.encode_board(
            game_state.board, game_state.current_player
        )
        
        # 获取合法动作
        valid_moves = game_state.get_valid_moves()
        valid_actions = self.action_encoder.encode_move_list(valid_moves)
        action_mask = self.action_encoder.create_action_mask(valid_moves)
        
        # 神经网络预测
        policy_probs, value = self.neural_network.predict(board_encoded, action_mask)
        
        # 展开节点
        if not node.is_expanded and len(valid_actions) > 0:
            node.expand(policy_probs, valid_actions)
        
        return value
    
    def _backup_path(self, path: List[MCTSNode], value: float):
        """
        沿路径回传价值
        
        Args:
            path: 节点路径
            value: 叶子节点价值
        """
        for i, node in enumerate(reversed(path)):
            # 交替取反价值（对手视角）
            node_value = value if i % 2 == 0 else -value
            node.visit_count += 1
            node.value_sum += node_value
    
    def get_best_action(self, game_state: GameState) -> Tuple[Position, Position]:
        """
        获取最佳动作
        
        Args:
            game_state: 当前游戏状态
            
        Returns:
            最佳移动 (起始位置, 目标位置)
        """
        action_probs, _ = self.search(game_state)
        
        # 选择概率最高的动作
        best_action_idx = np.argmax(action_probs)
        best_move = self.action_encoder.decode_action(best_action_idx)
        
        return best_move
    
    def get_training_data(self, game_state: GameState) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        获取训练数据
        
        Args:
            game_state: 当前游戏状态
            
        Returns:
            (棋盘编码, 动作概率分布, 价值评估)
        """
        # 执行搜索
        action_probs, root_value = self.search(game_state)
        
        # 编码棋盘状态
        board_encoded = self.board_encoder.encode_board(
            game_state.board, game_state.current_player
        )
        
        return board_encoded, action_probs, root_value


def test_mcts():
    """测试MCTS功能"""
    from .network import create_chess_net
    
    print("测试MCTS...")
    
    # 创建组件
    neural_net = create_chess_net()
    board_encoder = BoardEncoder()
    action_encoder = ActionEncoder()
    
    # 创建MCTS
    mcts = MCTS(
        neural_network=neural_net,
        board_encoder=board_encoder,
        action_encoder=action_encoder,
        num_simulations=100  # 测试用较少的模拟次数
    )
    
    # 创建游戏状态
    game_state = GameState()
    
    # 执行搜索
    print("执行MCTS搜索...")
    action_probs, root_value = mcts.search(game_state)
    
    print(f"动作概率分布形状: {action_probs.shape}")
    print(f"非零概率数量: {np.count_nonzero(action_probs)}")
    print(f"根节点价值: {root_value:.3f}")
    
    # 获取最佳动作
    best_move = mcts.get_best_action(game_state)
    print(f"最佳移动: {best_move[0]} -> {best_move[1]}")
    
    # 获取训练数据
    board_encoded, train_probs, train_value = mcts.get_training_data(game_state)
    print(f"训练数据 - 棋盘形状: {board_encoded.shape}")
    print(f"训练数据 - 概率形状: {train_probs.shape}")
    print(f"训练数据 - 价值: {train_value:.3f}")
    
    print("MCTS测试完成！")


if __name__ == "__main__":
    test_mcts()