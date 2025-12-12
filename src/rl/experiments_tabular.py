import gymnasium as gym
from rl.environment_blackjack import BlackjackEnvironment
from rl.environment import Environment
from rl.qlt import QLearningAgentTabular

def run_blackjack():
    env = BlackjackEnvironment(gym.make("Blackjack-v1", sab=True, natural=True))
    agent = QLearningAgentTabular(
        env=env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_decay_rate=0.001,
        min_epsilon=0.01,
        max_epsilon=1.0,
        verbose=True
    )
    history = agent.train(num_episodes=10000)
    print("Treinamento Blackjack finalizado!")

def run_cliff_walking():
    env = Environment(gym.make("CliffWalking-v1"))
    agent = QLearningAgentTabular(
        env=env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_decay_rate=0.001,
        verbose=True
    )
    agent.train(num_episodes=5000)
    print("Treinamento Cliff Walking finalizado!")

def run_frozen_lake():
    env = Environment(gym.make("FrozenLake-v1", is_slippery=True))
    agent = QLearningAgentTabular(
        env=env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_decay_rate=0.001,
        verbose=True
    )
    agent.train(num_episodes=10000)
    print("Treinamento Frozen Lake finalizado!")

if __name__ == "__main__":
    print("Escolha um ambiente: 1=Blackjack, 2=CliffWalking, 3=FrozenLake")
    choice = int(input("> "))
    if choice == 1:
        run_blackjack()
    elif choice == 2:
        run_cliff_walking()
    elif choice == 3:
        run_frozen_lake()
    else:
        print("Opção inválida.")
