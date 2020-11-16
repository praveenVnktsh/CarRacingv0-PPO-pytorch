
import gym
import cv2
import numpy as np
from config import Args

class Env():
    def __init__(self, args:Args):
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
        distances, _ = self.preprocess(img_rgb)
        self.stack = distances * self.args.valueStackSize  # four frames for decision
        # print('ENVIRONMENT RESET -- Stack size = ',self.stack )
        return np.array(self.stack)
    def checkGreen(self, img_rgb):
        _, gray = self.preprocess(img_rgb)
        temp = gray[73:93, 44:51]
        if temp.mean() < 100:
            return True
        return False

    def step(self, action, steps):
        finalReward = 0
        death = False
        rgbState = None
        reason = 'NULL'
        for i in range(self.args.action_repeat):
            rgbState, reward, envDeath, _ = self.env.step(action)
            
            if self.checkGreen(rgbState):
                finalReward -= 0.05
            finalReward += reward
            self.storeRewards(reward)
            death = True
            if self.checkExtendedPenalty():
                reason = 'Greenery'
                finalReward -= 10
            elif steps > self.args.deathThreshold:
                reason = 'Timesteps exceeded'
            # elif (envDeath and steps < int(1000/self.args.action_repeat) ):
            #     reason = 'Environment Signal'
            else:
                death = False
            if death:
                break
            

        distances, _ = self.preprocess(rgbState)
        
        for i in range(5):
            self.stack.pop(0)
        
        self.stack += distances

        assert len(self.stack) == self.args.valueStackSize * 5
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
        gray = cv2.resize(gray[0:83, 0:95], (96, 96))
        temp =  cv2.resize(gray.copy()[0:83, 0:95], (96, 96))
        temprgb = rgb.copy()[0:83, 0:95]
        

        x = 48
        y = 78

        locs = [None, None, None, None, None]
        for i in range(0, 96, 1):
            if None not in locs:
                break
            chk = (min(max(0, y-i), 95), min(max(x-i, 0), 95) )
            if locs[0] == None and self.checkPixelGreen(temp[chk]):
                locs[0] = chk
                cv2.circle(temprgb, (x - i + 1, y - i), 1, (255, 0, 0), 1) #leftmost

            chk = (min(max(0, y-i), 95), max(min(x+i, 95), 0) )
            if locs[4] == None and self.checkPixelGreen(temp[chk]): # rightmost
                locs[4] = chk
                cv2.circle(temprgb, (x + i - 1, y - i), 1, (255, 0, 0), 1)
            
            chk = (min(max(0, y-i), 95), x)
            if locs[2] == None and self.checkPixelGreen(temp[chk]): #middle
                locs[2] = chk
                cv2.circle(temprgb, (x, y - i), 1, (255, 0, 0), 1)

            chk = (min(max(0, y-i), 95), max(min(x + i//2, 95), 0))
            if locs[3] == None and self.checkPixelGreen(temp[chk]): #midright
                locs[3] = chk
                cv2.circle(temprgb, (x + i//2 - 1, y - i), 1, (255, 0, 0), 1)

            chk = (min(max(0, y-i), 95), min(max(x - i//2, 0), 95))
            if locs[1] == None and self.checkPixelGreen(temp[chk]): #midleft
                locs[1] = chk
                cv2.circle(temprgb, (x - i//2 + 1, y - i), 1, (255, 0, 0), 1)
        distances = []
        for i in range(len(locs)):
            if locs[i] == None:
                locs[i] = (self.args.maxDistance, self.args.maxDistance)
            dist = round(np.linalg.norm(np.array(locs[i]) - np.array((y, x))), 2)
            if dist == 0:
                dist = self.args.maxDistance
            distances.append(dist)


        temprgb =  cv2.resize(temprgb, (0,0), fx = 2, fy = 2)
        cv2.imshow('img', cv2.resize(temprgb, (300, 300)))
        return distances, gray

    def checkExtendedPenalty(self):
        temp = np.array(self.rewards)
        if temp[temp < 0].size == temp.size:
            return True
        return False
   
    def storeRewards(self, reward):
        if len(self.rewards) > self.args.deathByGreeneryThreshold:
            self.rewards.pop(0)
        self.rewards.append(reward)
        
        