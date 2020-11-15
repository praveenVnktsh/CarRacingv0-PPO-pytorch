import argparse
from comet_ml import Experiment
import gc
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
import matplotlib.pyplot as plt
import torch



parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=2, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--deathThreshold', type=int, default=2000, metavar='N', help='Threshold before death')
parser.add_argument('--saveLocation', type=str, default='model/new/', metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--clip-param', type=float, default=0.2, metavar='N', help='random seed (default: 0)')
parser.add_argument('--ppo-epoch', type=int, default=10, metavar='N', help='')
parser.add_argument('--buffer-capacity', type=int, default=1000, metavar='N', help='')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='')
args = parser.parse_args()




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)




if __name__ == "__main__":
    oldEpisodeIndex = 557
    newEpisodeIndex = 0
    agent = Agent(oldEpisodeIndex, args, device)
    env = Env(args)
    prevState = env.reset()

    for episodeIndex in range(oldEpisodeIndex, 100000):
        
        newEpisodeIndex = episodeIndex
        score = 0
        prevState = env.reset()
        die = False
        for t in range(10000):
            if t%200 - 1 == 0:
                gc.collect()
            action, a_logp = agent.select_action(prevState)
            curState, reward, done = env.step(action* np.array([-2., 1.0, 0.5]) + np.array([1., 0, 0.]), t)
            env.render()

            score += reward
            prevState = curState

            if (done or die):
                print("DEAD at score = ", score, t)
                break
        gc.collect()

        print('Ep {}\tLast score: {:.2f}'.format(episodeIndex, score))
        # logger('Ep {}\tLast score: {:.2f} {:.2f}'.format(episodeIndex, score))


   
