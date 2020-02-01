import numpy as np
import pandas as pd

import time

LENGTH = 20  # -----T the length
ACTIONS = ['left', 'right']
EPSILON = 1  # greedy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # the gamma
FRESH_TIME = 0.01
WAITING_TIME = 0.01
MAX_EPISODE = 100


def build_q_table(n, actions):
    return pd.DataFrame(
        np.zeros((n, len(actions))),
        columns=actions
    )


def choose_action(s, q_table, epsilon):
    all_action = q_table.iloc[s, :]
    if (np.random.uniform() > epsilon) or (all_action.all() == 0):
        action = np.random.choice(ACTIONS)
    else:
        action = all_action.argmax()
    return action


def get_reward(s, action):
    reward = 0
    s_ = s
    if action == 'right':
        s_ += 1
        if s_ == LENGTH - 1:
            s_ = 'terminal'
            reward = 1
    elif action == 'left':
        if s == 0:
            s_ = s
        else:
            s_ -= 1

    return s_, reward


def update_game(s, episode, step):
    print_list = ['-'] * (LENGTH - 1) + ['T']
    if s == 'terminal':
        desc = 'Episode %s: total_steps = %s' % (episode + 1, step)
        print(desc)
        time.sleep(WAITING_TIME)
    else:
        print_list[s] = 'o'
        desc = ''.join(print_list)
        print('\r{}'.format(desc), end='')
        time.sleep(FRESH_TIME)


def train():
    q_table = build_q_table(LENGTH, ACTIONS)
    for episode in range(MAX_EPISODE):
        step = 0
        s = 0
        is_terminated = False
        update_game(s, episode, step)

        while not is_terminated:
            action = choose_action(s, q_table, EPSILON)
            s_, reward = get_reward(s, action)
            q_predict = q_table.loc[s, action]
            if s_ != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[s_, :].max()
            else:
                q_target = reward
                is_terminated = True

            q_table.loc[s, action] += ALPHA * (q_target - q_predict)
            s = s_

            update_game(s, episode, step + 1)
            step += 1

    return q_table


if __name__ == "__main__":
    q_table = train()
    print(q_table)
