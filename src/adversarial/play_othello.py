"""
play_othello.py
Interface de linha de comando para Othello.
"""

import argparse
from board_othello import Board
from minimax_agent import choose_move as minimax_move
from mcts_agent import choose_move as mcts_move

def human_move(board, player):
    moves = board.legal_moves(player)
    if not moves:
        print("Sem movimentos — PASSA")
        return None

    board.print_board()
    print(f"Movimentos legais: {moves}")

    while True:
        cmd = input("Digite (linha coluna) ou 'pass': ")
        if cmd.strip().lower() == "pass":
            return None
        try:
            r, c = map(int, cmd.split())
            if (r, c) in moves:
                return (r, c)
        except:
            pass
        print("Movimento inválido.")

def play_game(mode, depth, iterations):
    board = Board()
    player = 1  # preto começa

    print("Tabuleiro inicial:")
    board.print_board()

    while not board.is_terminal():
        print(f"\nVez do jogador {'B' if player == 1 else 'W'}:")

        if mode == "human_vs_minimax":
            move = human_move(board, player) if player == 1 else minimax_move(board, player, depth)

        elif mode == "human_vs_mcts":
            move = human_move(board, player) if player == 1 else mcts_move(board, player, iterations)

        elif mode == "minimax_vs_mcts":
            move = minimax_move(board, player, depth) if player == 1 else mcts_move(board, player, iterations)

        elif mode == "mcts_vs_mcts":
            move = mcts_move(board, player, iterations)

        elif mode == "minimax_vs_minimax":
            move = minimax_move(board, player, depth)

        else:
            raise ValueError("Modo inválido.")

        print("Jogada:", move)
        board.apply_move(move, player)
        board.print_board()

        player = -player

    print("\nJogo terminado!")
    b, w = board.score()
    print(f"Placar final: B={b} | W={w}")
    print("Vencedor:", "Preto" if b > w else "Branco" if w > b else "Empate")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="human_vs_mcts")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    play_game(args.mode, args.depth, args.iterations)

if __name__ == "__main__":
    main()
