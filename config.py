import torch
import os


class Args():

    def __init__(self):
        clipper = 0.4
        trial = 1
        saveloc = 'model/distances/train_' + str(trial) + '_clip_' + str(clipper) + '/'

        self.gamma = 0.99
        self.action_repeat = 2
        self.valueStackSize = 4
        self.seed = 0

        self.numberOfLasers = 5
        self.deathThreshold = 2000
        self.clip_param = clipper
        self.saveLocation = saveloc
        self.ppo_epoch = 10
        self.buffer_capacity = 500
        self.batch_size = 128
        self.deathByGreeneryThreshold = 50
        self.maxDistance = 100
        

def configure():
    
    args = Args()
    os.makedirs(args.saveLocation, exist_ok = True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    return args, use_cuda, device