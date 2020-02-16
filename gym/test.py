import numpy as np
import gym

env = gym.make('CartPole-v0')
env.reset()
for i in range(1000):
    env.render()
    action = env.action_space.sample()
    print(action)
    env.step(action) # take a random action
    if i % 100 == 0:
        print("\r{}".format(i), end="")
env.close()