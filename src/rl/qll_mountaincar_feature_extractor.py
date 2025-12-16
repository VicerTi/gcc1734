import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

class Actions:
    LEFT = 0      # Acelerar para esquerda
    NONE = 1      # Não acelerar
    RIGHT = 2     # Acelerar para direita

class MountainCarFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.LEFT:  np.array([1, 0, 0]),
        Actions.NONE:  np.array([0, 1, 0]),
        Actions.RIGHT: np.array([0, 0, 1])
    }

    def __init__(self, env):
        self.env = env
        self.features_list = [
            self.f_bias,          # 1
            self.f_position,      # posição normalizada
            self.f_velocity,      # velocidade normalizada
            self.f_position_sq,   # posição²
            self.f_velocity_sq,   # velocidade²
            self.f_pos_action_interaction,  # posição x ação (específico)
            self.f_vel_action_interaction   # velocidade x ação
        ]

    def get_num_features(self):
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        return 3

    def get_action_one_hot_encoded(self, action):
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        # Terminal se posição >= 0.5 (goal)
        return state[0] >= 0.5

    def get_actions(self):
        return [Actions.LEFT, Actions.NONE, Actions.RIGHT]

    def get_features(self, state, action):
        feature_vector = np.zeros(len(self.features_list))
        for idx, f in enumerate(self.features_list):
            feature_vector[idx] = f(state, action)

        action_one_hot = self.get_action_one_hot_encoded(action)
        return np.concatenate([feature_vector, action_one_hot])

    # Features individuais
    def f_bias(self, state, action):
        return 1.0

    def f_position(self, state, action):
        # Normaliza posição [-1.2, 0.6] → [0, 1]
        pos = state[0]
        return (pos + 1.2) / 1.8

    def f_velocity(self, state, action):
        # Normaliza velocidade [-0.07, 0.07] → [0, 1]
        vel = state[1]
        return (vel + 0.07) / 0.14

    def f_position_sq(self, state, action):
        return self.f_position(state, action) ** 2

    def f_velocity_sq(self, state, action):
        return self.f_velocity(state, action) ** 2

    def f_pos_action_interaction(self, state, action):
        # Incentiva ações diferentes dependendo da posição
        pos_norm = self.f_position(state, action)
        if action == Actions.RIGHT:
            return pos_norm
        elif action == Actions.LEFT:
            return 1.0 - pos_norm
        return 0.0

    def f_vel_action_interaction(self, state, action):
        # Incentiva acelerar na direção da velocidade atual
        vel_norm = self.f_velocity(state, action)
        if action == Actions.RIGHT and state[1] > 0:
            return vel_norm
        elif action == Actions.LEFT and state[1] < 0:
            return 1.0 - vel_norm
        return 0.0