"""
中国象棋AI模块

包含神经网络模型、训练框架和推理引擎
"""

# 延迟导入，避免循环导入问题
__all__ = [
    'BoardEncoder',
    'ActionEncoder', 
    'ChessNet',
    'ChessTrainer',
    'MCTS'
]

def get_board_encoder():
    from .encoder import BoardEncoder
    return BoardEncoder

def get_action_encoder():
    from .encoder import ActionEncoder
    return ActionEncoder

def get_chess_net():
    from .network import ChessNet
    return ChessNet

def get_chess_trainer():
    from .trainer import ChessTrainer
    return ChessTrainer

def get_mcts():
    from .mcts import MCTS
    return MCTS