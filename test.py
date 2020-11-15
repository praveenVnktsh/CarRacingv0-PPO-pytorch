from comet_ml import Experiment
import gc
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
from config import configure




if __name__ == "__main__":
    args, use_cuda,  device = configure()
    checkpoint = 557
    agent = Agent(checkpoint, args, device)
    env = Env(args)

    for episodeIndex in range(checkpoint, 100000):
        score = 0
        prevState = env.reset()
        die = False
        for t in range(10000):
            action, a_logp = agent.select_action(prevState)
            curState, reward, done = env.step(action* np.array([-2., 1.0, 0.5]) + np.array([1., 0, 0.]), t)
            env.render()
            score += reward
            prevState = curState

            if (done or die):
                print("DEAD at score = ", score, t)
                break
        print('Ep {}\tLast score: {:.2f}\n----------------\n'.format(episodeIndex, score))


   
