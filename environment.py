import gym
import numpy as np


class Environment():
    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.stackNumber = 4
        self.rewardThresh = self.env.spec.reward_threshold

    def reset(self):
        self.dead = False
        self.counter = 0
        frame = self.rgb2gray(self.env.reset())

        self.currentFrameStack = [frame]*self.stackNumber
        self.currentFrameStacknp = np.array(self.currentFrameStack)
        return self.currentFrameStacknp

    def step(self, action):

        #removed loop to repeat actions for n steps

        frame, reward, dead, _ = self.env.step(action)
        if dead: #remove penalty from dead state
            reward += 100
        
        if np.mean(frame[:,:,1]) > 185.0:
            reward -= 0.05 #penalty if the person is in green tracks
        

        self.currentFrameStack.pop(0)
        self.currentFrameStack.append(self.rgb2gray(frame))
        self.currentFrameStacknp = np.array(self.currentFrameStack)

        
        return self.currentFrameStacknp, reward, dead
        


    def render(self, *arg):
        self.env.render(*arg)

    def rgb2gray(self, rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            gray = gray / 128. - 1.
        return gray
