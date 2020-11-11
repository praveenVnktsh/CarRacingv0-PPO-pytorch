
import gym
import cv2
import numpy as np
class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, args):
        self.env = gym.make('CarRacing-v0')
        self.args = args
        self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        processedImage = self.preprocess(img_rgb)
        self.stack = [processedImage] * self.args.img_stack  # four frames for decision
        return np.array(self.stack)
    def checkDeath(self, img_rgb):
        gray = self.preprocess(img_rgb)
        temp = gray[73:93, 44:51]
        # cv2.imshow('car', temp)
        # print(temp.mean())
        if temp.mean() < 100:
            return True
        return False
    def step(self, action, steps):
        total_reward = 0
        death = False
        for i in range(self.args.action_repeat):
            img_rgb, reward, envDeath, _ = self.env.step(action)

            
            total_reward += reward
            if envDeath:
                death = True
            

        img_gray = self.preprocess(img_rgb)
        cv2.imshow('img', cv2.resize(img_gray, (300, 300)))
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.args.img_stack
        return np.array(self.stack), total_reward, death

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def preprocess(rgb, norm=True):
        # rgb image -> gray [0, 1]
        # gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        # if norm:
        #     gray = gray / 128. - 1.

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        # print(gray.shape, type(gray), gray.dtype)
        # cv2.imshow('cut', )
        gray = cv2.resize(gray[0:83, 0:95], (96, 96))
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

