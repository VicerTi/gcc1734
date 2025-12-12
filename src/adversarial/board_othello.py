"""
board_othello.py
Implementação do jogo Othello (Reversi).
"""

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1)
]

class Board:
    def __init__(self, size=8):
        self.size = size
        self.grid = [[0] * size for _ in range(size)]
        mid = size // 2
        # Inicial clássico do Othello
        self.grid[mid-1][mid-1] = -1
        self.grid[mid-1][mid]   = 1
        self.grid[mid][mid-1]   = 1
        self.grid[mid][mid]     = -1

    def copy(self):
        b = Board(self.size)
        b.grid = [row[:] for row in self.grid]
        return b

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def _captures_in_dir(self, r, c, dr, dc, player):
        r += dr; c += dc
        captured = []
        while self.in_bounds(r, c) and self.grid[r][c] == -player:
            captured.append((r, c))
            r += dr; c += dc
        if self.in_bounds(r, c) and self.grid[r][c] == player and captured:
            return captured
        return []

    def legal_moves(self, player):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] != 0: 
                    continue
                for dr, dc in DIRECTIONS:
                    if self._captures_in_dir(r, c, dr, dc, player):
                        moves.append((r, c))
                        break
        return moves

    def apply_move(self, move, player):
        if move is None:
            return
        r, c = move
        flips = []
        for dr, dc in DIRECTIONS:
            flips += self._captures_in_dir(r, c, dr, dc, player)
        if not flips:
            raise ValueError(f"Jogada ilegal: {move}")
        self.grid[r][c] = player
        for fr, fc in flips:
            self.grid[fr][fc] = player

    def is_terminal(self):
        return not self.legal_moves(1) and not self.legal_moves(-1)

    def score(self):
        blacks = sum(v == 1 for row in self.grid for v in row)
        whites = sum(v == -1 for row in self.grid for v in row)
        return blacks, whites

    def winner(self):
        b, w = self.score()
        return 1 if b > w else -1 if w > b else 0

    def print_board(self):
        print("  " + " ".join(str(i) for i in range(self.size)))
        for i, row in enumerate(self.grid):
            line = []
            for val in row:
                if val == 1: line.append("B")
                elif val == -1: line.append("W")
                else: line.append(".")
            print(f"{i} " + " ".join(line))

    def count(self, player):
        return sum(1 for row in self.grid for v in row if v == player)
