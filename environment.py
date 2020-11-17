
from agentFile import Agent
import gym
import cv2
import numpy as np
from config import Args

class Env():
    def __init__(self, configs:Args):
        self.env = gym.make('CarRacing-v0')
        self.configs = configs
        self.env.seed(configs.seed)
        self.previousRewards = []
        self.rewardThresh = self.env.spec.reward_threshold
        self.rewards = []

    def reset(self):
        self.counter = 0
        self.rewards = []
        self.die = False
        img_rgb = self.env.reset()
        distances, _ = self.preprocess(img_rgb)
        self.stack = distances * self.configs.valueStackSize  + [0., 0., 0.]*self.configs.actionStack # four frames for decision
        # print('ENVIRONMENT RESET -- Stack size = ',self.stack )
        assert len(self.stack) == self.configs.valueStackSize * self.configs.numberOfLasers  + 3*self.configs.actionStack
        return np.array(self.stack)
   
    def checkGreen(self, img_rgb):
        _, gray = self.preprocess(img_rgb)
        temp = gray[66:78, 44:52]
        if temp.mean() < 100:
            return True
        return False

    def step(self, action, steps, agent:Agent):
        finalReward = 0
        death = False
        rgbState = None
        reason = 'NULL'
        for i in range(self.configs.action_repeat):
            rgbState, reward, envDeath, _ = self.env.step(self.configs.actionTransformation(action))
            
            self.env.render()
            if self.checkGreen(rgbState):
                finalReward -= 0.05
            
            jerkPenalty = 10*np.linalg.norm(np.array(agent.buffer['a'][agent.counter-1]) - np.array(agent.buffer['a'][agent.counter-2]))
            finalReward -= jerkPenalty
            finalReward -= action[2]


        
            finalReward += reward
            self.storeRewards(reward)
        
        
        
            death = True
            if self.checkExtendedPenalty():
                reason = 'Greenery'
                finalReward -= 10
            elif steps > self.configs.deathThreshold:
                reason = 'Timesteps exceeded'
            else:
                death = False
            if death:
                break
            

        distances, _ = self.preprocess(rgbState)
        
        self.stack = self.stack[self.configs.numberOfLasers:self.configs.valueStackSize * self.configs.numberOfLasers]
        self.stack += distances
        # self.stack += action.tolist()

        assert len(self.stack) == self.configs.valueStackSize * self.configs.numberOfLasers  + 3*self.configs.actionStack
        return np.array(self.stack), finalReward, death, reason

    def render(self, *arg):
        self.env.render(*arg)

    
    def checkPixelGreen(self, pixel):
        if pixel == 0:
            return True
        return False

    def preprocess(self, rgb):
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        gray = gray[0:83, 0:95]
        temp =  gray.copy()[0:83, 0:95] 
        temprgb = rgb.copy()[0:83, 0:95]
        

        x = 48
        y = 73

        locs = [None, None, None, None, None]
        for i in range(0, 95):
            if None not in locs:
                break
            chk = (min(max(0, y-i), 82), min(max(x-i, 0), 94) )
            if locs[0] == None and self.checkPixelGreen(temp[chk]):
                locs[0] = chk
                cv2.circle(temprgb, (x - i + 1, y - i), 1, (255, 0, 0), 1) #leftmost

            chk = (min(max(0, y-i), 82), max(min(x+i, 94), 0) )
            if locs[4] == None and self.checkPixelGreen(temp[chk]): # rightmost
                locs[4] = chk
                cv2.circle(temprgb, (x + i - 1, y - i), 1, (255, 0, 0), 1)
            
            chk = (min(max(0, y-i), 82), x)
            if locs[2] == None and self.checkPixelGreen(temp[chk]): #middle
                locs[2] = chk
                cv2.circle(temprgb, (x, y - i), 1, (255, 0, 0), 1)

            chk = (min(max(0, y-i), 82), max(min(x + i//2, 94), 0))
            if locs[3] == None and self.checkPixelGreen(temp[chk]): #midright
                locs[3] = chk
                cv2.circle(temprgb, (x + i//2 - 1, y - i), 1, (255, 0, 0), 1)

            chk = (min(max(0, y-i), 82), min(max(x - i//2, 0), 94))
            if locs[1] == None and self.checkPixelGreen(temp[chk]): #midleft
                locs[1] = chk
                cv2.circle(temprgb, (x - i//2 + 1, y - i), 1, (255, 0, 0), 1)
        
        
        distances = []
        for i in range(len(locs)):
            if locs[i] == None:
                locs[i] = (self.configs.maxDistance, self.configs.maxDistance)
            dist = round(np.linalg.norm(np.array(locs[i]) - np.array((y, x))), 2)
            if dist == 0:
                dist = self.configs.maxDistance
            distances.append(dist)


        temprgb =  cv2.resize(temprgb, (0,0), fx = 2, fy = 2)
        cv2.imshow('img', cv2.resize(temprgb, (300, 300)))
        # cv2.waitKey(200)
        return distances, gray

    def checkExtendedPenalty(self):
        temp = np.array(self.rewards)
        if temp[temp < 0].size == temp.size:
            return True
        return False
   

    def storeRewards(self, reward):
        if len(self.rewards) > self.configs.deathByGreeneryThreshold:
            self.rewards.pop(0)
        self.rewards.append(reward)
        
        