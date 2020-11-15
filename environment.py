
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
        self.previousRewards = []
        self.rewardThresh = self.env.spec.reward_threshold
        self.rewards = []


    def reset(self):
        self.counter = 0
        self.rewards = []
        self.die = False
        img_rgb = self.env.reset()
        processedImage = self.preprocess(img_rgb)
        self.stack = [processedImage] * self.args.img_stack  # four frames for decision
        return np.array(self.stack)
    def checkGreen(self, img_rgb):
        gray = self.preprocess(img_rgb)
        temp = gray[73:93, 44:51]
        if temp.mean() < 100:
            return True
        return False

    def step(self, action, steps):
        finalReward = 0
        death = False
        rgbState = None

        for i in range(self.args.action_repeat):
            rgbState, reward, envDeath, _ = self.env.step(action)
            
            if self.checkGreen(rgbState):
                finalReward -= 0.05
            finalReward += reward
            self.storeRewards(reward)
            death = self.checkExtendedPenalty() or steps > self.args.deathThreshold or (envDeath and steps < int(1000/self.args.action_repeat) )
            if death:
                break
            

        img_gray = self.preprocess(rgbState)
        cv2.imshow('img', cv2.resize(img_gray, (300, 300)))
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.args.img_stack
        return np.array(self.stack), finalReward, death

    def render(self, *arg):
        self.env.render(*arg)

    def preprocess(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        gray = cv2.resize(gray[0:83, 0:95], (96, 96))
        return gray

    def checkExtendedPenalty(self):
        temp = np.array(self.rewards)
        if temp[temp < 0].size == temp.size:
            return True
        return False
   
    def storeRewards(self, reward):
        if len(self.rewards) > 50:
            self.rewards.pop(0)
        self.rewards.append(reward)
        
        