import argparse
import torch
import os

def configure():
    clipper = 0.4
    saveloc = 'model/clip' + str(clipper) + '/'

    parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--action-repeat', type=int, default=2, metavar='N', help='repeat action in N frames (default: 8)')
    parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
    parser.add_argument('--deathThreshold', type=int, default=2000, metavar='N', help='Threshold before death')
    parser.add_argument('--clip-param', type=float, default=clipper, metavar='N', help='random seed (default: 0)')
    parser.add_argument('--saveLocation', type=str, default=saveloc, metavar='N', help='interval between training status logs (default: 10)')
    parser.add_argument('--ppo-epoch', type=int, default=10, metavar='N', help='')
    parser.add_argument('--buffer-capacity', type=int, default=1000, metavar='N', help='')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='')
    parser.add_argument('--deathByGreeneryThreshold', type=int, default=75, metavar='N', help='')
    args = parser.parse_args()

    os.makedirs(args.saveLocation, exist_ok = True)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    return args, use_cuda, device