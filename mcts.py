import torch
import numpy as np
import math

class MCTS:
    def __init__(self, env):
        self.env = env

    def sample_action(self, root_node, iteration, policy_network, value_network, sample_method = 'sample'):
        for i in range(iteration):
            self.simulation(root_node, policy_network, value_network)

        # print('root',root_node.visit)
        pi_mcts = root_node.get_children_distribution()
        node = root_node.get_max_visit_child()
        sampled_action = np.random.choice(np.arange(len(pi_mcts)), p=pi_mcts.numpy())
        # print('샘플',sampled_action)
        if sample_method == 'sample':
            return sampled_action, pi_mcts
        elif sample_method == 'max_visit':
            return node.action, pi_mcts
    
    def simulation(self, root_node,policy_network, value_network):
        history = []
        node = root_node
        history.append(node)

        while not node.is_terminal():
            node = node.select(self.env)
            history.append(node)
        
        logit, pi = policy_network(node.s.unsqueeze(0) if node.current_player ==1 else node.s.unsqueeze(0) * -1)
        valid_action_mask = self.env.get_action_mask(node.s)
        node.pi = pi.detach() * valid_action_mask
        # print(pi, logit,valid_action_mask,node.pi)
        node.expand()
        v = value_network(node.s.unsqueeze(0) if node.current_player ==1 else node.s.unsqueeze(0) * -1).detach()
        if node.done == True:
            v = node.winner
        self.backup(history, v)

    def backup(self, history, reward):
        for node in history[::-1]:
            node.u = node.p * (math.sqrt(node.parent.visit + 1) if node.parent != None else 1)/(1+node.visit)
            node.q = node.q + (reward - node.q)/(node.visit + 1)
            # node.q = ((node.q*node.visit) + reward)/(node.visit + 1)
            node.visit += 1 # n = visit



class Node:
    def __init__(self, p, current_player, action = None, parent = None):
        self.q = 0 # q <- q + 1/n * (v - q)
        self.p = p # p_st-1a
        self.pi = None # policy distribution
        self.visit = 0
        self.u = 0 # updated in backup step, u <- p/(1+n)
        self.children = []
        self.parent = parent
        self.action = action
        self.s = None
        self.done = False
        self.winner = None
        self.current_player = current_player # it means player to action in this state.

    def get_children_distribution(self):
        sum_visit = 0
        for child in self.children:
            sum_visit += child.visit

        pi_mcts = torch.zeros_like(self.pi[0], dtype=torch.float32)
        for child in self.children:
            pi_mcts[child.action] = child.visit / sum_visit
        
        return pi_mcts
    
    def get_max_visit_child(self):
        selectedNode = None
        visit = 0
        for child in self.children:
            if child.visit > visit:
                selectedNode = child
            # print('action :',child.action//5,child.action%5,'visit:',child.visit)

        return selectedNode
    
    def select(self, env):
        # select action along maximizing q + u, u = c_puct * p_sa * sqrt(parent_visit) / (1 + visit_sa)
        selectedNode = None
        z = -999
        for child in self.children:

            if child.q + child.u > z:
                selectedNode = child
                z = child.q + child.u
            # print('q :',child.q,'u :',child.u, child.q + child.u, child.q + child.u > -99)
        s = self.s.clone().detach()
        s[0, selectedNode.action//5, selectedNode.action%5] = self.current_player
        s[1] = selectedNode.current_player
        selectedNode.s = s
        # print(s.shape,s)
        result = env.check_win(s.numpy(), self.current_player) # True : win, None: draw, False: Not ended
        if result != False:
            selectedNode.done = True
            if result == True:
                selectedNode.winner = self.current_player
            else:
                selectedNode.winner = self.current_player/2 # draw
        return selectedNode
    
    def expand(self):
        if self.done:
            return
        for action, p in enumerate(self.pi[0]):
            if p.item() == 0:
                continue
            new_node = Node(p, self.current_player * -1, action=action, parent=self)
            new_node.u = p # initial u
            self.children.append(new_node)

    def is_terminal(self):
        return len(self.children) == 0 or self.done