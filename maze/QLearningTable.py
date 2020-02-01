import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, s):
        self.check_and_add_state(s)
        if np.random.uniform() < self.epsilon:
            all_actions = self.q_table.loc[s, :]
            action = np.random.choice(all_actions[all_actions == np.max(all_actions)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def train(self, s, action, reward, s_):
        self.check_and_add_state(s_)
        q_predict = self.q_table.loc[s, action]
        if s_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = reward
        self.q_table.loc[s, action] += self.learning_rate * (q_target - q_predict)

    def check_and_add_state(self, s):
        if s not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=s,
                )
            )
