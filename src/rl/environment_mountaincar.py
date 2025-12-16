import numpy as np
from rl.environment import Environment

class MountainCarEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)
        self.num_bins = 20  # Número de bins por dimensão (posição e velocidade)
        
        # Bordas dos bins para posição [-1.2, 0.6]
        low, high = self.env.observation_space.low, self.env.observation_space.high
        self.pos_bins = np.linspace(low[0], high[0], self.num_bins + 1)
        
        # Bordas dos bins para velocidade [-0.07, 0.07]
        self.vel_bins = np.linspace(low[1], high[1], self.num_bins + 1)

    def get_num_states(self):
        return self.num_bins ** 2  # 20 x 20 = 400 estados discretos

    def get_num_actions(self):
        return self.env.action_space.n  # 3 ações: esquerda (0), nada (1), direita (2)

    # Código principal da discretização
    def get_state_id(self, state):
        pos, vel = state
        # Discretiza posição e velocidade
        pos_bin = np.digitize(pos, self.pos_bins) - 1
        vel_bin = np.digitize(vel, self.vel_bins) - 1
        # Cria 20 bins uniformes para posição e velocidade / Atribui o bin correspondente a cada valor contínuo
        
        # Clipa para evitar índices fora do range e garantir índices válidos (0 a 19)
        pos_bin = np.clip(pos_bin, 0, self.num_bins - 1)
        vel_bin = np.clip(vel_bin, 0, self.num_bins - 1)
        
        # Mapeia para ID único: pos_bin * num_bins + vel_bin => 20 x 20 = 400 estados
        return pos_bin * self.num_bins + vel_bin
        # Converte o par (pos_bin, vel_bin) em um único ID inteiro

    def get_random_action(self):
        return self.env.action_space.sample()