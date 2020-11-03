
from torch import device, dist
from neuralnet import NeuralNet
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

class Agent():


    def __init__(self,gamma = 0.99,  clipValue = 0.1, epochs = 10, maxGradNorm = 0.5, lr = 0.001, stackNumber = 4, bufferLength = 2000, batchSize = 128 ):

        self.bufferLength = bufferLength
        self.batchSize = batchSize

        

        self.clipValue = clipValue
        self.epochs = epochs
        self.maxGradNorm = maxGradNorm


        self.gamma = gamma
        self.trainingSteps = 0
        self.counter = 0

        self.net = NeuralNet().double()
        print(self.net)
  
        self.optimizer = optim.Adam(self.net.parameters(), lr = lr) 


        transition = np.dtype([('state', np.float64, (stackNumber, 96, 96)), ('action', np.float64, (3,)), ('actionLogP', np.float64),
                       ('reward', np.float64), ('state_', np.float64, (stackNumber, 96, 96))])
        self.buffer = np.empty(self.bufferLength, dtype=transition)
        

    def chooseAction(self, state):
        state = torch.from_numpy(state).double().unsqueeze(0)

        with torch.no_grad():
            print(state.shape)

            alpha, beta = self.net(state)[0] #compute outputs of NN

        distribution = Beta(alpha, beta) #find the distribution

        action = distribution.sample() #pick an action

        actionLogProbability = distribution.log_prob(action).sum(dim = 1).item() #find the probability of that action

        action  = action.squeeze().cpu().numpy()

        return action, actionLogProbability

    def saveModel(self):
        torch.save(self.net.state_dict(), 'model/ppo.pkl')


    def saveToBuffer(self, tranisition):
        self.buffer[self.counter] = tranisition
        self.counter += 1
        if self.counter == self.bufferCapacity:
            self.counter = 0
            return True
        return False


    def update(self):

        self.trainingSteps += 1

        #define the sarsa (state action reward state action .....)


        state = torch.tensor(self.buffer['state'], dtype = torch.double).to(device)
        action = torch.tensor(self.buffer['action'], dtype = torch.double).to(device)
        reward = torch.tensor(self.buffer['reward'], dtype = torch.double).to(device).view(-1, 1)
        state_ = torch.tensor(self.buffer['state_'], dtype = torch.double).to(device)


        prevActionLogP = torch.tensor(self.buffer['actionLogP'], dtype = torch.double).to(device).view(-1, 1)


        with torch.no_grad():
            targetValue = reward + self.gamma*self.net(state_)[1]

            advance = targetValue - self.net(state)[1]


        for i in range(self.epochs):

            for index in BatchSampler(SubsetRandomSampler(range(self.bufferLength)), self.batchSize, False ):
                

                alpha, beta = self.net(state[index])[0]

                distribution = Beta(alpha, beta)

                actionLogP = distribution.log_prob(action[index]).sum(dim = 1, keepdim = True)

                ratio = torch.exp(actionLogP - prevActionLogP[index])

                surr1 = ratio*advance[index]

                surr2 = torch.clamp(ratio, 1.0 - self.clipValue, 1.0 + self.clipValue)*advance[index]

                actionLoss = -torch.min(surr1, surr2).mean()

                valueLoss = Functional.smooth_l1_loss(self.net(state[index])[1], targetValue[index])


                loss = actionLoss + 2.0*valueLoss

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()




                


