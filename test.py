from comet_ml import Experiment
import gc
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
from config import configure




if __name__ == "__main__":
    configs, use_cuda,  device = configure()
    
    agent = Agent(configs.checkpoint, configs, device)
    env = Env(configs)

    for episodeIndex in range(configs.checkpoint, 100000):
        score = 0
        prevState = env.reset()
        die = False
        for t in range(10000):
            action, a_logp = agent.select_action(prevState)
            curState, reward, done, reason = env.step(configs.actionTransformation(action), t, agent)
            env.render()


            score += reward
            prevState = curState

            if done:
                print('--------------------')
                print("Dead at score = ", round(score, 2), ' || Timesteps = ', t, ' || Reason = ', reason)
                break
        print('Ep {}\tLast score: {:.2f}\n----------------\n'.format(episodeIndex, score))


   
