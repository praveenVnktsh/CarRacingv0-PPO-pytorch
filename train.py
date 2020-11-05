import argparse
import gc
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
import matplotlib.pyplot as plt
import torch


parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)





def logger(string, fileName = 'log.txt' ):
    file = open(fileName, 'a')
    file.write(string)
    file.write('\n')
    file.close()


def beginPlot():
    global fig, ax
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    
    fig.show()
    fig.canvas.draw()

def plot(data):
    global ax, fig
    ax.clear()
    ax.plot(data)
    ax.set_ylim(bottom=0)
    fig.canvas.draw()
    # `start_event_loop` is required for console, not jupyter notebooks.
    # Don't use `plt.pause` because it steals focus and makes it hard
    # to stop the app.
    fig.canvas.start_event_loop(0.001)




if __name__ == "__main__":
    
    # beginPlot()
    
    i_ep_old = 0
    test = False

    scores = [0]
    runningscores = [0]
    i_ep_new = 0
    running_score = 0
    

    while i_ep_old != 100000 - 1:
        agent = Agent(i_ep_old)
        env = Env(args)
        state = env.reset()
        try:
            
            for i_ep in range(i_ep_old, 100000):
                
                i_ep_new = i_ep
                score = 0
                state = env.reset()
                die = False
                # while not die:
                for t in range(10000):
                    if t%200 - 1 == 0:
                        gc.collect()
                    action, a_logp = agent.select_action(state)
                    # state_, reward, done, die = env.step(env.action_space.sample() * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                    # state_, reward, done = env.step([0.0, 1.0, 0.5 ], t)
                    state_, reward, done = env.step(action* np.array([-2., 1.0, 0.5]) + np.array([1., 0, 0.]), t)
                    # if test:
                    env.render()
                    # sleep(0.03)
                    if not test:
                        agent.update((state, action, a_logp, reward, state_))

                    score += reward
                    state = state_

                    if (done or die) and not test:
                        print("DEAD at score = ", score, t)
                        break
                running_score = running_score * 0.99 + score * 0.01
                gc.collect()
                scores.append(score)
                runningscores.append(runningscores)
                
                # plot(scores)

                print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f} \n'.format(i_ep, score, running_score))
                logger('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))


                if i_ep % args.log_interval == 0 and not test:
                    agent.save_param(i_ep)
                # if running_score > env.reward_threshold:
                #     print("Solved! Running reward is now {} and the last episode runs to {}!\n".format(running_score, score))
                #     break
        except Exception as e:
            i_ep_old = i_ep_new
            i_ep_old = (i_ep_old//args.log_interval)*args.log_interval
            logger('ENV RESTARTING')
            env.env.close()
            del agent
            del env
            print(e)
