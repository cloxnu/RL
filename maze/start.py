from View import Maze
from QLearningTable import QLearningTable

def update():
    for episode in range(100):
        s = maze.reset()

        while True:
            maze.render()

            action = RL.choose_action(str(s))
            s_, reward, is_done = maze.step(action)
            RL.train(str(s), action, reward, str(s_))
            s = s_

            if is_done:
                break


if __name__ == '__main__':
    maze = Maze()
    RL = QLearningTable(actions=list(range(maze.num_action)))

    maze.after(100, update)
    maze.mainloop()