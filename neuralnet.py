import torch.nn as nn

class NeuralNet(nn.Module): #actor critic network 

    def __init__(self, config):

        super(NeuralNet, self).__init__()

        self.baseNetwork = nn.Sequential(
            nn.Conv2d(config['stackNumber'], 8, kernel_size = 4, stride= 2), # stacked frames as input
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2),
            nn.ReLU(),
        ) #this outputs a (256, 1, 1) tensor


        self.valueNetwork = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
        )

        self.alphaNetwork  = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softplus(),
        )

        self.betaNetwork = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softplus(),
        )
    
        self.apply(self.initializeWeights())

    def initializeWeights(self, module):

        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight, gain = nn.init.calculate_gain('relu')) #initialize weights
            nn.init.constant_(module.bias, 0.1) #set bias to 0.1 constant
        
    def forward(self, input):

        out = self.baseNetwork(input)
        out = out.view(-1, 256) #resize inputs for next outputs

        value = self.valueNetwork(out)
        alpha = self.alphaNetwork(out) + 1
        beta = self.alphaNetwork(out) + 1

        return (alpha, beta), value