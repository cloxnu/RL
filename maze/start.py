from View import Maze
from QLearningTable import QLearningTable

def update():
    for episode in range(100):
        s = env.reset()