from neuralnet import Net
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 32

    def __init__(self, episode, args, device):

        transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                    ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])

        self.args = args
        self.training_step = 0
        self.net = Net(args).double().to(device)
        self.device = device
        if episode != 0:
            print("LOADING FROM EPISODE", episode)
            self.net.load_state_dict(torch.load(self.args.saveLocation + 'episode-' + str(episode) + '.pkl'))
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self, episode ):
        print('-----------------------------------------')
        print("SAVING AT EPISODE", episode)
        print('-----------------------------------------')
        torch.save(self.net.state_dict(), self.args.saveLocation + 'episode-' + str(episode) +  '.pkl')

        

    def update(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        # print('COUNTER = ', self.counter)
        if self.counter == self.buffer_capacity:
            print("UPDATING WEIGHTS at reward = ", transition[3])
            self.counter = 0
            self.training_step += 1

            s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
            a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
            r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
            s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

            old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

            with torch.no_grad():
                target_v = r + self.args.gamma * self.net(s_)[1]
                advantage = target_v - self.net(s)[1]
                # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(self.ppo_epoch):
                for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                    alpha, beta = self.net(s[index])[0]
                    dist = Beta(alpha, beta)
                    a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                    ratio = torch.exp(a_logp - old_a_logp[index])

                    surr1 = ratio * advantage[index]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage[index]
                    actorLoss = -torch.min(surr1, surr2).mean()
                    criticLoss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                    loss = actorLoss + 2. * criticLoss

                    self.optimizer.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                    self.optimizer.step()