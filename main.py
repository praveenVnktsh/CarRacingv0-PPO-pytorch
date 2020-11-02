from agentFile import Agent
import os 
import torch 
import gym
import re
import sys

TRAINCONFIG = {

    'totalEpisodes': 10000
}


AGENTCONFIG = {
    'min_epsilon':0.1,
    'num_frame_stack':4,
    'epsion_decay_steps':100000,
    'gamma':0.95, #discount rate
}

env_name = 'CarRacing-v0'
env = gym.make(env_name)

agent = Agent(AGENTCONFIG)


def playOneEpisode(render = False):
    reward, frames = agent.playEpisode(render)
    print('Episode - ', agent.episodeCount, ' Reward - ', reward, 'Frames - ', frames, 'Total Steps = ', agent.steps)


def saveCheckpoint():
    pass


def train():

    while agent.episodeCount < TRAINCONFIG['totalEpisodes']:

        if agent.episodeCount % 100 == 0:
            render = True
            saveCheckpoint()
        else:
            render = False

        playOneEpisode(render)

    print('Training Complete ------------')    



if __name__ == "__main__":
    train()