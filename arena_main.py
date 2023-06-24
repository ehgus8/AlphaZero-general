import torch
from game import TicTacToe
from replayBuffer import ReplayBuffer
from mcts import MCTS, Node
from network import ActorCritic
import torch.optim as optim
from arena import Arena

def main():
    env = TicTacToe()
    recent_model = ActorCritic(in_channels=2,policy_out=5*5)
    enemy_model = ActorCritic(in_channels=2,policy_out=5*5)
    path = './models/'
    recent_model.load_state_dict(torch.load(path + 'model_state_dict_cnt117.pt'))
    enemy_model.load_state_dict(torch.load(path + 'model_state_dict_cnt17.pt'))

    arena = Arena(env)
    arena.set_enemy_model(enemy_model)
    arena.set_recent_model(recent_model)
    print("승률 :",arena.play(50,0.55))

if __name__ == '__main__':
    main()