from  RL.tut.maze_env import Maze
from RL.tut.RL_brain import QLearningTable

EPISODES=100

def update():
    for epsiode in range(EPISODES):
        #initial obervation
        obervation=env.reset()

        while True:
            #fresh env
            env.render()

            ##RL choose action based on obervation
            action=RL.choose_action(str(obervation))

            ##RL take action and next observation and reward
            obervation_,reward,done=env.step(action)

            ##RL learn from this transition
            RL.learn(str(obervation),action,reward,str(obervation_))

            ##swap obervation
            obervation=obervation_

            ##break while loop when end of this episode
            if done:
                break
    ##end of game
    print('game over')
    env.destroy()

if __name__=='__main__':
    env=Maze()
    RL=QLearningTable(actions=list(range(env.n_actions)))

    env.after(100,update)
    env.mainloop()







