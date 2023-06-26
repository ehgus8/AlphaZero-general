from network import ActorCritic
from replayBuffer import ReplayBuffer
from connect4 import Connect4

import torch
import torch.optim as optim

def main():
    input_channel = 1
    selected_game = Connect4
    env = selected_game()
    memory = ReplayBuffer(150000)
    memory.load('./memories/memory_n_episode_5600.pkl')
    game_size = env.size()
    model = ActorCritic(in_channels=input_channel, policy_out=game_size*game_size, game_size=game_size)
    path = './models/'
    model.load_state_dict(torch.load(path + 'model_state_dict_cnt51.pt'))
    epoch = 30
    batch_size = 128
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train_net(memory, optimizer, batch_size=batch_size, epoch=epoch)
    torch.save(model.state_dict(), './models/' + f'model_state_dict_cnt{51}_epoch30_older.pt')
    print('model saved!')

if __name__ == '__main__':
    main()