import numpy as np
import pandas as pd


class RL:
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
        pass

    def check_and_add_state(self, s):
        if s not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=s,
                )
            )


class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy)

    def train(self, s, action, reward, s_):
        self.check_and_add_state(s_)
        q_predict = self.q_table.loc[s, action]
        if s_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = reward
        self.q_table.loc[s, action] += self.learning_rate * (q_target - q_predict)


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy)

    def train(self, s, action, reward, s_, action_):
        self.check_and_add_state(s_)
        q_predict = self.q_table.loc[s, action]
        if s_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[s_, action_]
        else:
            q_target = reward
        self.q_table.loc[s, action] += self.learning_rate * (q_target - q_predict)


class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_and_add_state(self, s):
        if s not in self.q_table.index:
            addition = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=s,
            )
            self.q_table = self.q_table.append(addition)
            self.eligibility_trace = self.eligibility_trace.append(addition)

    def train(self, s, action, reward, s_, action_):
        self.check_and_add_state(s_)
        q_predict = self.q_table.loc[s, action]
        if s_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[s_, action_]
        else:
            q_target = reward

        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, action] = 1

        self.q_table += self.learning_rate * (q_target - q_predict) * self.eligibility_trace

        self.eligibility_trace *= self.gamma * self.lambda_

