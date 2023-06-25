import torch
import numpy as np

from game import TicTacToe
from connect4 import Connect4
from replayBuffer import ReplayBuffer
from mcts import MCTS, Node
from network import ActorCritic
import torch.optim as optim
from arena import Arena

def main():
    input_channel = 1
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    # env = TicTacToe()
    selected_game = Connect4
    env = selected_game()
    memory = ReplayBuffer(100000)
    memory.load('./memories/memory_n_episode_200.pkl')
    mcts = MCTS(env)
    game_size = env.size()
    model = ActorCritic(in_channels=input_channel, policy_out=game_size*game_size, game_size=game_size)
    
    epoch = 1
    batch_size = 64
    train_interval = 100
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_cnt = 0 # the number of training
    arena = Arena(selected_game())

    win, draw, lose = 0, 0, 0
    for n_epi in range(1000000):
        s, _ = env.reset()
        done = False
        tmp_memory = []
        root_node = Node(p=0, current_player=1)
        root_node.s = torch.from_numpy(s).float().detach()
        current_player = 1
        while not done:
            a, pi, node = mcts.sample_action(root_node=root_node, 
                                       iteration = 50,
                                       model = model,
                                       sample_method='mcts_distribution')
            s_prime, r, done = env.step(a)
            turn = current_player
            tmp_memory.append([s, pi, r, turn])
            s = s_prime
            current_player *= -1
            root_node = node
            # if current_player == -1:
            #     root_node.s = torch.from_numpy(env.get_reversed_board(s)).float().detach()
            # else:
            #     root_node.s = torch.from_numpy(s).float().detach()
        
        if r == 1:
            win+=1
        elif r == -1:
            lose += 1
        else:
            draw += 1

        print(f"n_episode :{n_epi}")
        env.display_board()
        for history in tmp_memory:
            s, pi, _, turn = history
            tmp_reward = r
            if r == 1: # if winner is 1
                if turn == -1:
                    s = env.get_reversed_board(s)
                    tmp_reward = -1
            elif r == -1: # if winner is -1
                if turn == -1:
                    s = env.get_reversed_board(s)
                    tmp_reward = 1
            history[0] = s
            history[2] = tmp_reward
            # print(history)
        
        memory.add(tmp_memory)
        if n_epi % 200 == 0 and n_epi != 0:
            memory.save(f'./memories/memory_n_episode_{n_epi}.pkl')

        if memory.size() > batch_size and n_epi % train_interval == 0:
            print(f'무승부 비율 : {draw/(win+draw+lose) * 100}%')
            win, draw, lose = 0, 0, 0
            train_cnt += 1
            prev_model = ActorCritic(in_channels=input_channel,policy_out=game_size*game_size, game_size=game_size)
            prev_model.load_state_dict(model.state_dict())
            model.train_net(memory, optimizer, batch_size=batch_size, epoch=epoch)
            arena.set_enemy_model(prev_model)
            arena.set_recent_model(model)
            result = arena.play(n=60, threshold=0.55)
            print('승률 :',result, 'memory len :', memory.size())
            if result < 0.55:
                model = prev_model
            else:
                torch.save(model.state_dict(), './models/' + f'model_state_dict_cnt{train_cnt}.pt')
                print('model save!', 'cnt :', train_cnt)
        
        # if train_cnt % 2 == 0 and train_cnt != 0:
        #     torch.save(model.state_dict(), './models/' + f'model_state_dict_cnt{train_cnt}.pt')
        # if n_epi % print_interval == 0 and n_epi != 0:
        #     print(f"n_episode :{n_epi}")

if __name__ == '__main__':
    main()