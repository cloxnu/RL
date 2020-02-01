from View import Maze
from RL import SarsaLambdaTable

def update():
    for episode in range(100):
        s = maze.reset()

        action = RL.choose_action(str(s))

        RL.eligibility_trace *= 0

        while True:
            maze.render()

            s_, reward, is_done = maze.step(action)
            action_ = RL.choose_action(str(s_))
            RL.train(str(s), action, reward, str(s_), action_)
            s = s_
            action = action_

            if is_done:
                break


if __name__ == '__main__':
    maze = Maze()
    RL = SarsaLambdaTable(actions=list(range(maze.num_action)))

    maze.after(100, update)
    maze.mainloop()