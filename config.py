import torch
import os
import numpy as np


class Args():

    def __init__(self):
        self.checkpoint = 622
        trial = 5
        self.test = False


        self.gamma = 0.99
        self.action_repeat = 4
        self.valueStackSize = 8
        self.actionStack = 0
        self.seed = 0

        self.numberOfLasers = 5
        self.deathThreshold = 2000
        self.clip_param = 0.4
        
        self.ppo_epoch = 10
        self.buffer_capacity = 500
        self.batch_size = 128
        self.deathByGreeneryThreshold = 35
        self.maxDistance = 100

        self.actionMultiplier = np.array([2., 1.0, 1.0])
        self.actionBias = np.array([-1.0, 0.0, 0.0])

        
        saveloc = 'model/distances/train_' + str(trial) + '/'


        self.saveLocation = saveloc

        os.makedirs(self.saveLocation, exist_ok = True)
        f = open(saveloc + 'params.json','w')
        f.write(str(self.getParamsDict()))
        f.close()



    
    def getParamsDict(self):
        ret = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        print('\nHYPERPARAMETERS = ', ret)
        return ret
    
    def actionTransformation(self, action):
        return action*self.actionMultiplier + self.actionBias
        

def configure():
    
    args = Args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    return args, use_cuda, device