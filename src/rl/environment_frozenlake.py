from rl.environment import Environment

class FrozenLakeEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)

    def get_num_states(self):
        return self.env.observation_space.n  # 16 estados (4x4 grid)

    def get_num_actions(self):
        return self.env.action_space.n  # 4 ações (left, down, right, up)

    def get_state_id(self, state):
        return state  # Já é um inteiro discreto

    def get_random_action(self):
        return self.env.action_space.sample()