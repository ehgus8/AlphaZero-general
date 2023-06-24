import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, in_channels, policy_out):
        super(ActorCritic, self).__init__()
        self.epoch = 0
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pi = nn.Linear(16*5*5, policy_out)
        self.v = nn.Linear(16*5*5, 1)

    def policy(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        pi = self.pi(x)
        prob = F.softmax(pi,1)
        return pi, prob

    def value(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        v = F.tanh(self.v(x))
        return v

    def train_net(self, memory, optimizer, batch_size):
        iter = memory.size()//batch_size + 1
        epoch_loss = 0
        for i in range(iter):
            s, pi, r = memory.sample(batch_size)
            # print(s.shape, pi.shape, r.shape)
            # print(self.value(s), self.value(s).shape)
            loss =  F.mse_loss(self.value(s), r) + F.cross_entropy(self.policy(s)[0], pi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += s.shape[0] * loss.item()

        # print epoch loss
        print(f"epoch :{self.epoch + 1}, loss :{epoch_loss / memory.size()}")