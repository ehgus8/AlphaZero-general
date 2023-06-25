import numpy as np
import torch

size = 5

class Connect4:
    def __init__(self):
        self.current_player = 1
        # self.valid_moves = np.ones(5*5)
    
    def size(self):
        return size

    def reset(self):
        self.board = np.zeros((1, size, size),dtype=np.float32)
        self.current_player = 1
        # self.board[1] = self.current_player
        self.valid_moves = np.ones(size*size)
        s_prime = self.board.copy()
        r_dummy = 0

        return s_prime, r_dummy
    
    def step(self, action):
        y, x = action//size, action%size
        done = self.make_move(self.board, y, x, self.current_player)
        self.current_player *= -1
        # self.board[1] = self.current_player
        s_prime = self.board.copy()
        reward = 0
        if done == True:
            if self.winner == 0:
                reward = 0
                print('드로우드로우드로우!!!')
            elif self.current_player == -1:
                reward = 1
            else:
                reward = -1
                
        return s_prime, reward, done

    def get_action_mask(self, s):
        mask = s[0].clone().flatten()
        mask[mask == 0] = 2
        mask[(mask == 1) | (mask == -1)] = 0
        mask[mask == 2] = 1
    
        return mask
        
    # 플레이어 1은 -1 , 플레이어 2는 1로 변경 된 상태를 넘겨준다.
    def get_reversed_board(self, s):
        return s * -1
    
    def get_state(self):
        return self.board.copy()

    def get_random_move(self):
        valid_indices = np.array(np.where(self.valid_moves == 1)).reshape(-1)
        random_move = np.random.choice(valid_indices,size=1)
        return random_move[0]
    
    
    def get_valid_moves(self, board):
        indices = np.where(board == 0)
        valid_moves = indices[0] * 5 + indices[1]
        return valid_moves

    def make_move(self, board, row, col, player):
        if board[0,row, col] != 0:
            return None

        board[0, row, col] = player
        self.valid_moves[row * size + col] = 0
        
        is_win = self.check_win(board, player)
        if is_win:
            print(f'Player {player} wins!')
            self.winner = player
            return True
        #draw
        if is_win == None:
            print(f'Player draw!')
            self.winner = 0
            return True
        
        return False

    def check_win(self, board, player):
        # Check rows and columns
        for i in range(size):
            for j in range(size - 3):
                if (np.all(board[0, i, j:j+4] == player) or 
                    np.all(board[0, j:j+4, i] == player)):
                    return True

        diag1 = [1, 1] # vector (y, x)
        diag2 = [1, -1] # vector (y, x)
        # Check diagonals
        for i in range(size - 3):
            for j in range(size - 3):
                if (np.all(board[0, [i+k*diag1[0] for k in range(4)], [j+k*diag1[1] for k in range(4)]] == player) or
                    np.all(board[0, [i+k*diag2[0] for k in range(4)], [j+2+k*diag2[1] for k in range(4)]] == player)):
                    return True

        # Check draw
        if (np.where(board == 0)[0].shape[0] == 0):
            return None

        return False

    def display_board(self):
        print(self.board[0])

    