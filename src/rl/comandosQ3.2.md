python -m rl.ql_train --agent tabular --env_name MountainCar-v0 --num_episodes 20000 --learning_rate 0.1 --gamma 0.99 --min_epsilon 0.01 --seed 42 --plot

python -m rl.ql_play --agent tabular --env_name MountainCar-v0 --num_episodes 10 --seed 42