from View import Maze
from DQN import DQN


def update():
    step = 0
    for episode in range(300):
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1

    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = DQN(env.num_action, 
             env.num_feature,
             learning_rate=0.01,
             reward_decay=0.9,
             e_greedy=0.9,
             replace_target_iter=10,
             memory_size=4000,
             )

    env.after(100, update)
    env.mainloop()
