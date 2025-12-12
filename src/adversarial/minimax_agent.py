"""
minimax_agent.py
Implementação do agente Minimax com heurística.
"""

import math

WEIGHTS = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100]
]

def heuristic(board, player):
    my_count = board.count(player)
    opp_count = board.count(-player)

    if my_count + opp_count == 0:
        piece_diff = 0
    else:
        piece_diff = 100 * (my_count - opp_count) / (my_count + opp_count)

    pos = 0
    for r in range(board.size):
        for c in range(board.size):
            if board.grid[r][c] == player:
                pos += WEIGHTS[r][c]
            elif board.grid[r][c] == -player:
                pos -= WEIGHTS[r][c]

    my_moves = len(board.legal_moves(player))
    opp_moves = len(board.legal_moves(-player))

    if my_moves + opp_moves == 0:
        mobility = 0
    else:
        mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

    return pos * 0.6 + piece_diff * 0.2 + mobility * 0.2

def minimax(board, depth, player, max_player, alpha=-math.inf, beta=math.inf):
    if depth == 0 or board.is_terminal():
        return heuristic(board, max_player), None

    moves = board.legal_moves(player)
    if not moves:
        return minimax(board, depth-1, -player, max_player, alpha, beta)[0], None

    best_move = None

    if player == max_player:
        value = -math.inf
        for move in moves:
            b2 = board.copy()
            b2.apply_move(move, player)
            score = minimax(b2, depth-1, -player, max_player, alpha, beta)[0]
            if score > value:
                value = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = math.inf
        for move in moves:
            b2 = board.copy()
            b2.apply_move(move, player)
            score = minimax(b2, depth-1, -player, max_player, alpha, beta)[0]
            if score < value:
                value = score
                best_move = move
            beta = min(beta, score)
            if alpha >= beta:
                break
        return value, best_move

def choose_move(board, player, depth=3):
    return minimax(board, depth, player, player)[1]
