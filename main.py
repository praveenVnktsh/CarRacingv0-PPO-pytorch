from agentFile import Agent
from environment import Environment
import numpy as np


def startTraining(agent, env, render = False, logInterval = 1000):

    trainingRecords = []
    runningScore = 0

    for episodeCount in range(100000):
        score = 0
        state = env.reset()

        for t in range(1000):

            action, actionLogP = agent.chooseAction(state)

            state_, reward, done, dead = env.step(action*np.array([2.0, 1.0, 1.0] + np.array([-1.0, 0.0, 0.0])))
            

            if render:
                env.render()

            if agent.store((state, action, actionLogP, reward, state_ )):
                print("UPDATING")
                agent.update()

            score += reward
            state = state_
            if done or dead:
                break
       
        runningScore = runningScore*0.99 + score*0.01

        render = False
        if episodeCount % logInterval == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(episodeCount, score, runningScore))
            agent.save_param()
            render = True


        

        if runningScore > env.rewardThresh:
            print("SOLVED MAN!!!!")
            break
    




if __name__ == "__main__":
    agent = Agent()
    env = Environment()

    startTraining(agent, env)


