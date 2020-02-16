import numpy as np
import gym

env = gym.make('CartPole-v0')
env.reset()

action = 0
for i in range(1000):
    env.render()
    # action = env.action_space.sample()
    if action == 1:
        action = 0
    else:
        action = 1
    observation, reward, done, info = env.step(action) # take a random action
    # print(observation.shape)
    # observation = observation.reshape(-1, 4)
    # print(observation.shape)
    print("ob: {} | reward: {} | done: {} | info: {}".format(observation, reward, done, info))
    if i % 50 == 0:
        print("\r{}".format(i), end="")
env.close()