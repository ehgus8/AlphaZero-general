from connect4 import Connect4
from mcts import MCTS, Node
from network import ActorCritic

import torch

def play(model, env, mcts, turn, sample_method = 'mcts_distribution'):
    
    root_node = Node(p=0, current_player=turn)
    root_node.s = torch.from_numpy(env.get_state()).float().detach()

    a, pi, node = mcts.sample_action(root_node=root_node, 
                                iteration = 400,
                                model = model,
                                sample_method=sample_method)
    
    print(pi)
    s_prime, r, done = env.step(a)

    if done == True:
        return True
    
    return False

def main():
    input_channel = 1
    selected_game = Connect4
    env = selected_game()
    mcts = MCTS(env)
    game_size = env.size()
    model = ActorCritic(in_channels=input_channel, policy_out=game_size*game_size, game_size=game_size)
    path = './models/'
    model.load_state_dict(torch.load(path + 'model_state_dict_cnt110_epoch30.pt'))

    s, _ = env.reset()
    done = False

    human_turn = int(input("Select turn 1: you first, -1: agent first"))

        
    turn = 1
    while not done:
        if turn == human_turn:
            env.display_board()
            action = list(map(int,input("coordinate(y x): ").split()))
            action = action[0] * game_size + action[1]
            s_prime, r, done = env.step(action=action)
            turn = turn * -1
        if done:
            break

        done = play(model, env, mcts, turn, sample_method='mcts_distribution')
        turn = turn * -1
    env.display_board()

if __name__ == '__main__':
    main()