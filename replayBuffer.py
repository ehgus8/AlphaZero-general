import collections
import random
import numpy as np
import torch
import pickle
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def add(self, transitions):
        self.buffer.extend(transitions)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, p_lst, r_lst = [], [], []

        for transition in mini_batch:
            s, p, r = transition
            s_lst.append(s)
            p_lst.append(p)
            r_lst.append([r])
        # print(s_lst)
        # print(torch.tensor(s_lst, dtype=torch.float32),torch.tensor(s_lst, dtype=torch.float32).shape)
        # print(torch.stack(p_lst),torch.stack(p_lst).shape)

        return torch.tensor(np.array(s_lst), dtype=torch.float32), \
                torch.stack(p_lst), \
                torch.tensor(r_lst, dtype=torch.float32)
    
    def size(self):
        return len(self.buffer)
    
    def save(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self.buffer, fp)

    def load(self, path):
        with open(path, 'rb') as fp:
            self.buffer = pickle.load(fp)