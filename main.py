from agentFile import Agent
import gym
from environment import Environment
from neuralnet import NeuralNet
TRAINCONFIG = {

    'totalEpisodes': 10000
}


AGENTCONFIG = {
    'min_epsilon':0.1,
    'num_frame_stack':4,
    'epsion_decay_steps':100000,
    'gamma':0.95, #discount rate
}




def playOneEpisode(agent, render = False):
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

        playOneEpisode(agent, render)

    print('--------------Training Complete ------------')    



if __name__ == "__main__":


    env_name = 'CarRacing-v0'
    env = gym.make(env_name)
    agent = Agent(AGENTCONFIG)
    train(agent, env)