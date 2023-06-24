import torch
from game import TicTacToe
from replayBuffer import ReplayBuffer
from mcts import MCTS, Node
from network import ActorCritic
import torch.optim as optim
from arena import Arena

def main():
    input_channel = 2
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    env = TicTacToe()
    memory = ReplayBuffer(200000)
    mcts = MCTS(env)
    model = ActorCritic(in_channels=input_channel,policy_out=5*5)
    
    print_interval = 100
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_cnt = 0 # the number of training
    arena = Arena(TicTacToe())

    for n_epi in range(1000000):
        s, _ = env.reset()
        done = False
        tmp_memory = []
        root_node = Node(p=0, current_player=1)
        root_node.s = torch.from_numpy(s).float()
        current_player = 1
        while not done:
            a, pi = mcts.sample_action(root_node=root_node, 
                                       iteration = 25,
                                       policy_network = model.policy,
                                       value_network = model.value,
                                       sample_method='mcts_distribution')
            s_prime, r, done = env.step(a)
            tmp_memory.append([s, pi, r])
            s = s_prime
            current_player *= -1
            root_node = Node(p=0, current_player=1)
            if current_player == -1:
                root_node.s = torch.from_numpy(env.get_reversed_board(s)).float()
            else:
                root_node.s = torch.from_numpy(s).float()
        
        print(f"n_episode :{n_epi}")
        env.display_board()
        for history in tmp_memory:
            history[2] = r
        
        memory.add(tmp_memory)

        if memory.size() > 512 and n_epi % print_interval == 0:
            train_cnt += 1
            prev_model = ActorCritic(in_channels=input_channel,policy_out=5*5)
            prev_model.load_state_dict(model.state_dict())
            model.train_net(memory, optimizer, 512)
            arena.set_enemy_model(prev_model)
            arena.set_recent_model(model)
            result = arena.play(n=50, threshold=0.60)
            print('승률 :',result, 'memory len :', memory.size())
            if result < 0.60:
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