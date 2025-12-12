## Questão 2.2: Adversarial

1. # Código Principal a ser Rodado:

python play_othello.py

2. # Comandos Disponíveis:

### Jogador Humano vs MCTS
python play_othello.py --mode human_vs_mcts --iterations 500

### Jogador Humano vs Minimax
python play_othello.py --mode human_vs_minimax --depth 3

### Agente Minimax vs MCTS
python play_othello.py --mode minimax_vs_mcts --depth 3 --iterations 500

### Agente MCTS vs MCTS
python play_othello.py --mode mcts_vs_mcts --iterations 500

### Agente Minimax vs Minimax
python play_othello.py --mode minimax_vs_minimax --depth 4