python -m rl.ql_train --agent linear --env_name Blackjack-v1 --num_episodes 500000 --learning_rate 0.001 --gamma 1.0 --min_epsilon 0.05 --seed 42 --plot

python -m rl.ql_train --agent linear --env_name CliffWalking-v1 --num_episodes 5000 --learning_rate 0.1 --gamma 0.99 --min_epsilon 0.01 --seed 42 --plot

python -m rl.ql_train --agent linear --env_name FrozenLake-v1 --num_episodes 10000 --learning_rate 0.1 --gamma 0.95 --min_epsilon 0.01 --seed 42 --plot


python -m rl.ql_play --agent linear --env_name Blackjack-v1 --num_episodes 20 --seed 42

python -m rl.ql_play --agent linear --env_name CliffWalking-v1 --num_episodes 20 --seed 42

python -m rl.ql_play --agent linear --env_name FrozenLake-v1 --num_episodes 50 --seed 42