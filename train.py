import argparse
from comet_ml import Experiment
import gc
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
import matplotlib.pyplot as plt
import torch
import os


parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=2, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--deathThreshold', type=int, default=2000, metavar='N', help='Threshold before death')
parser.add_argument('--saveLocation', type=str, default='model/clip0x2/', metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--clip-param', type=float, default=0.2, metavar='N', help='random seed (default: 0)')
parser.add_argument('--ppo-epoch', type=int, default=10, metavar='N', help='')
parser.add_argument('--buffer-capacity', type=int, default=1000, metavar='N', help='')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='')

args = parser.parse_args()

os.makedirs(args.saveLocation, exist_ok = True)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


def logger(string, fileName = args.saveLocation + 'log.txt' ):
    file = open(fileName, 'a')
    file.write(string)
    file.write('\n')
    file.close()




if __name__ == "__main__":



    experiment = Experiment(project_name="CarRacing", api_key='P4Y69RtjtY1e0R20FCgvxtbi0' )
    hyper_params = {
        "gamma": args.gamma,
        "action-repeat": args.action_repeat,
        "img-stack":args.img_stack,
        "seed": args.seed,
        "clip_param" : args.clip_param,
        "ppo_epoch" : args.ppo_epoch,
        "buffer_capacity" : args.buffer_capacity,
        "batch_size" : args.batch_size,
        "log-interval": args.log_interval,
        "deathThreshold": args.deathThreshold,
        "saveLocation": args.saveLocation
    }
    experiment.log_parameters(hyper_params)
    
    oldEpisodeIndex = 0

    if oldEpisodeIndex == 0:
        f = open(args.saveLocation + 'log.txt','w')
        f.close()
    

    newEpisodeIndex = 0
    
    with experiment.train():
        while oldEpisodeIndex != 100000 - 1:
            agent = Agent(oldEpisodeIndex, args, device)
            env = Env(args)
            prevState = env.reset()
            try:
                
                for episodeIndex in range(oldEpisodeIndex, 100000):
                    
                    newEpisodeIndex = episodeIndex
                    score = 0
                    prevState = env.reset()
                    for t in range(10000):
                        if t%200 - 1 == 0:
                            gc.collect()
                        action, a_logp = agent.select_action(prevState)
                        curState, reward, done = env.step(action* np.array([-2., 1.0, 0.5]) + np.array([1., 0, 0.]), t)
                        env.render()

                        agent.update((prevState, action, a_logp, reward, curState), episodeIndex)

                        score += reward
                        prevState = curState

                        if done:
                            print("DEAD at score = ", score, t)
                            break
                    gc.collect()
                    experiment.log_metric("scores", score , step=episodeIndex)

                    print('Ep {}\tLast score: {:.2f}'.format(episodeIndex, score))
                    # logger('Ep {}\tLast score: {:.2f} {:.2f}'.format(episodeIndex, score))


            except Exception as e:
                oldEpisodeIndex = agent.lastSavedEpisode
                logger('ENV RESTARTING')
                env.env.close()
                del agent
                del env
                print(e)
