import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, in_channels, policy_out, game_size):
        super(ActorCritic, self).__init__()
        self.epoch = 0
        self.conv1 = nn.Conv2d(in_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pi = nn.Linear(64*game_size*game_size, policy_out)
        self.v1 = nn.Linear(64*game_size*game_size, policy_out)
        self.v2 = nn.Linear(policy_out, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        pi = self.policy(x)
        v = self.value(x)
        
        return pi, v

    def policy(self, x):
        # x = self.conv1(x)
        # x = F.relu(self.bn1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = torch.flatten(x, 1)
        pi = self.pi(x)
        prob = F.softmax(pi,1)
        return pi, prob

    def value(self, x):
        # x = self.conv1(x)
        # x = F.relu(self.bn1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = torch.flatten(x, 1)
        v = self.v1(x)
        v = F.tanh(self.v2(v))
        return v

    def train_net(self, memory, optimizer, batch_size, epoch = 1):
        iter = memory.size()//batch_size + 1
        self.to('cuda:0')
        for epoch in range(epoch):
            epoch_loss = 0
            epoch_val_loss = 0
            epoch_policy_loss = 0
            for i in range(iter):
                s, pi, r = memory.sample(batch_size)
                s = s.to('cuda:0')
                pi = pi.to('cuda:0')
                r = r.to('cuda:0')
                # print(s.shape, pi.shape, r.shape)
                # print(self.value(s), self.value(s).shape)
                pi_pred, v_pred = self(s)
                val_loss = F.mse_loss(v_pred, r)
                policy_loss = F.cross_entropy(pi_pred[0], pi)
                loss = val_loss + policy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_val_loss += s.shape[0] * val_loss.item()
                epoch_policy_loss += s.shape[0] * policy_loss.item()
                epoch_loss += s.shape[0] * loss.item()
            # print epoch loss
            print(f"epoch :{epoch + 1}, loss :{epoch_loss / memory.size()}, val_loss :{epoch_val_loss / memory.size()}, policy_loss :{epoch_policy_loss / memory.size()}")
        self.to('cpu')