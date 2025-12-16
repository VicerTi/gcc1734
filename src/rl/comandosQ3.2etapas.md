

# Etapa 1: Dividir a posição e velocidade em um número fixo de intervalos (bins)

Para transformar o espaço de estados contínuo em discreto, divide cada dimensão em **20 intervalos uniformes** (bins), resultando em **20 × 20 = 400 estados discretos** no total.

No construtor do wrapper `MountainCarEnvironment`:

```python
self.num_bins = 20

# Bordas dos bins para posição [-1.2, 0.6]
self.pos_bins = np.linspace(-1.2, 0.6, self.num_bins + 1)

# Bordas dos bins para velocidade [-0.07, 0.07]
self.vel_bins = np.linspace(-0.07, 0.07, self.num_bins + 1)
```

Essa divisão uniforme garante cobertura completa do espaço de observação do ambiente, conforme os limites oficiais do Gymnasium.

# Etapa 2: Mapear os estados contínuos para representações discretas

Implementa-se o mapeamento no método `get_state_id(self, state)`:

```python
def get_state_id(self, state):
    pos, vel = state
    
    # Atribui o bin correspondente a cada valor contínuo
    pos_bin = np.digitize(pos, self.pos_bins) - 1
    vel_bin = np.digitize(vel, self.vel_bins) - 1
    
    # Proteção contra valores exatamente nas bordas
    pos_bin = np.clip(pos_bin, 0, self.num_bins - 1)
    vel_bin = np.clip(vel_bin, 0, self.num_bins - 1)
    
    # Converte o par (pos_bin, vel_bin) em um único ID inteiro
    return pos_bin * self.num_bins + vel_bin
```

- `np.digitize` retorna o índice do bin em que o valor cai.
- O `clip` evita índices inválidos em casos extremos.
- O cálculo `pos_bin * 20 + vel_bin` cria um identificador único para cada combinação de bins, permitindo o uso direto da tabela Q tabular existente.

# Etapa 3: Adaptar o Q-Learning tabular para treinar o agente neste espaço discretizado

A adaptação foi mínima e totalmente contida no wrapper:
- O agente `QLearningAgentTabular` não precisou de nenhuma alteração, pois continua recebendo estados como inteiros discretos via `env.get_state_id(state)`.
- Os métodos obrigatórios da classe base `Environment` foram implementados:
  - `get_num_states()` -> retorna 400
  - `get_num_actions()` -> retorna 3
  - `get_random_action()` -> usa `env.action_space.sample()`

Comando de treinamento:
```
python -m rl.ql_train --agent tabular --env_name MountainCar-v0 --num_episodes 20000 --learning_rate 0.1 --gamma 0.99 --min_epsilon 0.01 --seed 42 --plot
```

