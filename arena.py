from mcts import MCTS, Node
import torch

class Arena:
    def __init__(self, env):
        self.env = env

    def set_enemy_model(self, model):
        self.enemy_model = model
    
    def set_recent_model(self, model):
        self.recent_model = model

    def play(self, n, threshold):
        mcts = MCTS(self.env)
        recent_model_win_cnt = 0
        enemy_model_win_cnt = 0
        for n_epi in range(n):
            s, _ = self.env.reset()
            done = False

            root_node = Node(p=0, current_player=1)
            root_node.s = torch.from_numpy(s).float()
            current_player = 1
            if n_epi % 2 == 0:
                models = [self.recent_model, self.enemy_model]
            else:
                models = [self.enemy_model, self.recent_model]
            turn = 0
            while not done:
                a, pi, node = mcts.sample_action(root_node=root_node, 
                                        iteration = 50,
                                        model = models[turn % 2],
                                        sample_method='mcts_distribution')
                s_prime, r, done = self.env.step(a)

                if done:
                    # if current_player == 1:
                    if r == 0:
                        recent_model_win_cnt += 0.5
                        enemy_model_win_cnt += 0.5
                        print('model draw')
                    elif self.recent_model == models[turn % 2]:
                        recent_model_win_cnt += 1
                        print('recent model win')
                    else:
                        enemy_model_win_cnt += 1      
                        print('recent model lose')

                s = s_prime
                current_player *= -1
                root_node = Node(p=0, current_player=1)
                if current_player == -1:
                    root_node.s = torch.from_numpy(self.env.get_reversed_board(s)).float()
                else:
                    root_node.s = torch.from_numpy(s).float()
                turn += 1
            if  enemy_model_win_cnt > n - n * threshold:
                print('실제 승률 :', recent_model_win_cnt / (recent_model_win_cnt + enemy_model_win_cnt))
                return -threshold
            if recent_model_win_cnt > n * threshold:
                print('실제 승률 :', recent_model_win_cnt / (recent_model_win_cnt + enemy_model_win_cnt))
                return threshold * 100
            print(f"n_episode :{n_epi}")
            self.env.display_board()

        return recent_model_win_cnt / (recent_model_win_cnt + enemy_model_win_cnt)
