import torch
from game import TicTacToe
from replayBuffer import ReplayBuffer
from mcts import MCTS, Node
from network import ActorCritic
import torch.optim as optim
from arena import Arena
from connect4 import Connect4

def main():
    selected_game = Connect4
    env = selected_game()
    game_size = env.size()
    recent_model = ActorCritic(in_channels=1,policy_out=game_size*game_size, game_size=game_size)
    enemy_model = ActorCritic(in_channels=1,policy_out=game_size*game_size, game_size=game_size)
    path = './models/'
    recent_model.load_state_dict(torch.load(path + 'model_state_dict_cnt1.pt'))
    enemy_model.load_state_dict(torch.load(path + 'model_state_dict_cnt110.pt'))

    arena = Arena(env)
    arena.set_enemy_model(enemy_model)
    arena.set_recent_model(recent_model)
    print("승률 :",arena.play(50,0.55))

if __name__ == '__main__':
    main()