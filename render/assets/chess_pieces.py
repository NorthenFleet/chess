"""
中国象棋棋子SVG图像资源
提供所有棋子的SVG矢量图像定义
"""

# 红方棋子SVG图像
RED_PIECES = {
    'king': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#ff4444" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#000">帅</text>
    </svg>''',
    
    'advisor': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#ff4444" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#000">仕</text>
    </svg>''',
    
    'elephant': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#ff4444" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#000">相</text>
    </svg>''',
    
    'horse': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#ff4444" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#000">马</text>
    </svg>''',
    
    'rook': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#ff4444" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#000">车</text>
    </svg>''',
    
    'cannon': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#ff4444" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#000">炮</text>
    </svg>''',
    
    'pawn': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#ff4444" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#000">兵</text>
    </svg>'''
}

# 黑方棋子SVG图像
BLACK_PIECES = {
    'king': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#333333" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#fff">将</text>
    </svg>''',
    
    'advisor': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#333333" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#fff">士</text>
    </svg>''',
    
    'elephant': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#333333" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#fff">象</text>
    </svg>''',
    
    'horse': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#333333" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#fff">马</text>
    </svg>''',
    
    'rook': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#333333" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#fff">车</text>
    </svg>''',
    
    'cannon': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#333333" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#fff">炮</text>
    </svg>''',
    
    'pawn': '''<svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="18" fill="#333333" stroke="#000" stroke-width="2"/>
        <text x="20" y="26" text-anchor="middle" font-family="SimHei" font-size="14" fill="#fff">卒</text>
    </svg>'''
}

def get_piece_svg(piece_type, color):
    """获取棋子的SVG图像"""
    if color == 'red':
        return RED_PIECES.get(piece_type, '')
    elif color == 'black':
        return BLACK_PIECES.get(piece_type, '')
    return ''

def get_all_pieces():
    """获取所有棋子图像"""
    return {
        'red': RED_PIECES,
        'black': BLACK_PIECES
    }