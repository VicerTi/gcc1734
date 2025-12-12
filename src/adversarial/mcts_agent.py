"""
mcts_agent.py
Monte Carlo Tree Search para Othello.
"""

import math, random

class Node:
    def __init__(self, board, player, parent=None, move=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self._untried = None

    def untried_moves(self):
        if self._untried is None:
            moves = self.board.legal_moves(self.player)
            self._untried = moves if moves else [None]
        return self._untried

    def best_child(self, c=1.4):
        return max(
            self.children,
            key=lambda ch: ch.wins / (ch.visits + 1e-9) + c * math.sqrt(math.log(self.visits + 1) / (ch.visits + 1e-9))
        )

def simulate(board, player):
    b = board.copy()
    p = player
    while not b.is_terminal():
        moves = b.legal_moves(p)
        move = random.choice(moves) if moves else None
        b.apply_move(move, p)
        p = -p
    return b.winner()

def backpropagate(node, winner):
    while node:
        node.visits += 1
        if winner == node.player:
            node.wins += 1
        node = node.parent

def mcts(board, player, iterations=500):
    root = Node(board.copy(), player)

    for _ in range(iterations):
        node = root

        # Seleção
        while node.untried_moves() == [] and node.children:
            node = node.best_child()

        # Expansão
        moves = node.untried_moves()
        if moves:
            move = moves.pop()
            b2 = node.board.copy()
            b2.apply_move(move, node.player)
            child = Node(b2, -node.player, parent=node, move=move)
            node.children.append(child)
            node = child

        # Simulação
        winner = simulate(node.board, node.player)

        # Retropropagação
        backpropagate(node, winner)

    # Melhor movimento
    if not root.children:
        return None
    return max(root.children, key=lambda ch: ch.visits).move

def choose_move(board, player, iterations=500):
    return mcts(board, player, iterations)
