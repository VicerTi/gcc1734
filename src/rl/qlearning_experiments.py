import gymnasium as gym
from rl.qlt import QLearningAgentTabular

def train_agent(env_name, lr, gamma, decay, episodes):
    env = gym.make(env_name)
    agent = QLearningAgentTabular(
        env=env,
        learning_rate=lr,
        gamma=gamma,
        epsilon_decay_rate=decay,
        max_epsilon=1.0,
        min_epsilon=0.01,
        verbose=False
    )
    history = agent.train(episodes)
    return agent, history

def test_agent(agent, env_name, episodes=100):
    env = gym.make(env_name)
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        state = agent.env.get_state_id(state)
        done = False
        total = 0
        while not done:
            action = agent.choose_action(state, is_in_exploration_mode=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.env.get_state_id(next_state)
            total += reward
            done = terminated or truncated
            state = next_state
        total_rewards.append(total)
    return sum(total_rewards) / len(total_rewards)

if __name__ == "__main__":
    configs = {
        "Blackjack-v1":  {"lr": 0.1, "gamma": 0.95, "decay": 0.001, "episodes": 200000},
        "CliffWalking-v0": {"lr": 0.1, "gamma": 0.90, "decay": 0.005, "episodes": 8000},
        "FrozenLake-v1": {"lr": 0.1, "gamma": 0.95, "decay": 0.003, "episodes": 20000},
    }

    for env_name, params in configs.items():
        print("\n========================================")
        print(f"Treinando agente no ambiente {env_name}...")
        agent, hist = train_agent(env_name, **params)
        avg_reward = test_agent(agent, env_name)
        print(f"→ Recompensa média no teste: {avg_reward:.3f}")
