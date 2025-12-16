import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

class Actions:
    UP    = 0
    RIGHT = 1
    DOWN  = 2
    LEFT  = 3

class CliffWalkingFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.UP:    np.array([1, 0, 0, 0]),
        Actions.RIGHT: np.array([0, 1, 0, 0]),
        Actions.DOWN:  np.array([0, 0, 1, 0]),
        Actions.LEFT:  np.array([0, 0, 0, 1])
    }

    def __init__(self, env):
        self.env = env
        self.features_list = [
            self.f0,  # bias
            self.f1,  # coluna normalizada (progresso horizontal até o goal)
            self.f2   # indicador se está na linha do cliff (linha 3)
        ]

    def get_num_features(self):
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        return 4

    def get_action_one_hot_encoded(self, action):
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        # Estados 37 a 46 = cliff, 47 = goal
        return state >= 37

    def get_actions(self):
        return [Actions.UP, Actions.RIGHT, Actions.DOWN, Actions.LEFT]

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
        """Coluna normalizada (0 a 11) — mede progresso em direção ao goal (coluna 11)"""
        col = state % 12
        return col / 11.0

    def f2(self, state, action):
        """Indicador se está na linha inferior (linha 3), próxima ao cliff"""
        row = state // 12
        return 1.0 if row == 3 else 0.0