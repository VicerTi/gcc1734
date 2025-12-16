import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

class Actions:
    LEFT  = 0
    DOWN  = 1
    RIGHT = 2
    UP    = 3

class FrozenLakeFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.LEFT:  np.array([1, 0, 0, 0]),
        Actions.DOWN:  np.array([0, 1, 0, 0]),
        Actions.RIGHT: np.array([0, 0, 1, 0]),
        Actions.UP:    np.array([0, 0, 0, 1])
    }

    def __init__(self, env):
        self.env = env
        self.grid_size = 4  # FrozenLake 4x4
        self.features_list = []
        self.features_list.append(self.f0)  # bias
        self.features_list.append(self.f1)  # distância Manhattan normalizada até o goal (15)

    def get_num_features(self):
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        return 4

    def get_action_one_hot_encoded(self, action):
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        # Goal (15) ou buraco (H) são terminais
        desc = self.env.unwrapped.desc.flatten()
        return desc[state] in [b'G', b'H']

    def get_actions(self):
        return [Actions.LEFT, Actions.DOWN, Actions.RIGHT, Actions.UP]

    def get_features(self, state, action):
        feature_vector = np.zeros(len(self.features_list))
        for index, feature in enumerate(self.features_list):
            feature_vector[index] = feature(state, action)
        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])
        return feature_vector

    def f0(self, state, action):
        """Bias term"""
        return 1.0

    def f1(self, state, action):
        """Distância Manhattan normalizada até o goal (posição 15)"""
        row = state // self.grid_size
        col = state % self.grid_size
        goal_row, goal_col = 3, 3  # goal está em (3,3) -> índice 15
        dist = abs(row - goal_row) + abs(col - goal_col)
        return 1.0 - (dist / 6.0)  # normaliza: 0 (longe) a 1 (próximo)
        # máximo possível é 6 (de (0,0) até (3,3))