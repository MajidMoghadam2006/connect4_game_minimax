import numpy as np

""" 
Player 1: max player
Player 2: min player
"""

def update_board(board, move, player_num):
    """update the game board to move to more depth in the ge tree"""
    if 0 in board[:, move]:
        update_row = -1
        for row in range(1, board.shape[0]):
            update_row = -1
            if board[row, move] > 0 and board[row - 1, move] == 0:
                update_row = row - 1
            elif row == board.shape[0] - 1 and board[row, move] == 0:
                update_row = row

            if update_row >= 0:
                board[update_row, move] = player_num
                break
    else:
        err = 'Invalid move by player {}. Column {}'.format(player_num, move)
        raise Exception(err)
    return board

def game_completed(board, player_num):
    """check if the player has won the game"""
    player_win_str = '{0}{0}{0}{0}'.format(player_num)
    to_str = lambda a: ''.join(a.astype(str))

    def check_horizontal(b):
        for row in b:
            if player_win_str in to_str(row):
                return True
        return False

    def check_vertical(b):
        return check_horizontal(b.T)

    def check_diagonal(b):
        for op in [None, np.fliplr]:
            op_board = op(b) if op else b

            root_diag = np.diagonal(op_board, offset=0).astype(np.int)
            if player_win_str in to_str(root_diag):
                return True

            for i in range(1, b.shape[1] - 3):
                for offset in [i, -i]:
                    diag = np.diagonal(op_board, offset=offset)
                    diag = to_str(diag.astype(np.int))
                    if player_win_str in diag:
                        return True

        return False

    return (check_horizontal(board) or
            check_vertical(board) or
            check_diagonal(board))

def terminal_state(board):
    """ Check who won the game and return the utility """
    if game_completed(board, player_num=1):
        return 100000
    elif game_completed(board, player_num=2):
        return -100000
    else:
        return None


def available_actions(board):
    """ Given the current game state returns the valid columns """
    valid_cols = []
    for col in range(board.shape[1]):
        if 0 in board[:, col]:
            valid_cols.append(col)
    return valid_cols


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def max_value(self, state, alpha, beta, depth):
        """max value calculation for alpha-beta Minimax algorithm"""
        action_values = [0 for _ in range(state.shape[1])]  # -2 is correct?????
        utility = terminal_state(state)
        if utility is not None:  # Game has a winner
            return utility, action_values
        avail_actions = available_actions(state)
        if len(avail_actions) == 0:  # game is tie
            return 0, action_values

        if depth == 0:
            return self.evaluation_function(state), action_values

        v = -float('inf')
        for a in avail_actions:
            state_ = update_board(state.copy(), a, player_num=1)  # next state
            v = max(v, self.min_value(state_, alpha, beta, depth-1))
            action_values[a] = v
            if v >= beta: return v, action_values
            alpha = max(alpha, v)
        return v, action_values

    def min_value(self, state, alpha, beta, depth):
        """min value calculation for alpha-beta Minimax algorithm"""
        utility = terminal_state(state)
        if utility is not None:  # Game has a winner
            return utility
        avail_actions = available_actions(state)
        if len(avail_actions) == 0:  # game is tie
            return 0

        if depth == 0:
            return -self.evaluation_function(state)

        v = +float('inf')
        for a in avail_actions:
            state_ = update_board(state.copy(), a, player_num=2)  # next state
            mxv, _ = self.max_value(state_, alpha, beta, depth-1)
            v = min(v, mxv)
            if v <= alpha: return v
            beta = min(beta, v)
        return v

    
    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        avail_actions = available_actions(board)
        # heuristic depth value -  a logarithmic function of # available actions
        n = len(avail_actions)
        if n == 6:
            d = 4
        elif n == 5:
            d = 5
        elif n == 4:
            d = 6
        elif n == 3:
            d = 8
        elif n == 2:
            d = 13
        else:
            d = 1
        '''Calculate action values using depth-limited heuristic-based minimax algorithm'''
        _, action_values = self.max_value(board, alpha=-float('inf'), beta=float('inf'), depth=d)
        print(action_values)
        '''Select best action from available actions based on action values returned from minimax'''
        best_action = avail_actions[0]
        best_value = -float('inf')
        for i in avail_actions:
            if action_values[i] > best_value:
                best_action = i
                best_value = action_values[i]
        return best_action

    '''max value calculation for Expectimax algorithm'''
    def max_value_exp(self, state, depth):

        action_values = [0 for _ in range(state.shape[1])]
        utility = terminal_state(state)
        if utility is not None:  # Game has a winner
            return utility, action_values
        avail_actions = available_actions(state)
        if len(avail_actions) == 0:  # game is tie
            return 0.0, action_values

        if depth == 0:
            return self.evaluation_function(state), action_values

        v = -float('inf')
        for a in avail_actions:
            state_ = update_board(state.copy(), a, player_num=1)  # next state
            v = max(v, self.exp_value(state_, depth-1))
            action_values[a] = v
        return v, action_values

    '''expectation value calculation for Expectimax algorithm'''
    def exp_value(self, state, depth):

        utility = terminal_state(state)
        if utility is not None:  # Game has a winner
            return utility
        avail_actions = available_actions(state)
        if len(avail_actions) == 0:  # game is tie
            return 0.0

        if depth == 0:
            return self.evaluation_function(state)

        v = 0.0
        for a in avail_actions:
            state_ = update_board(state.copy(), a, player_num=1)  # next state
            p = 1 / len(available_actions(state_))
            mxv, _ = self.max_value_exp(state_, depth - 1)
            v += p*max(v, mxv)
        return v

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        avail_actions = available_actions(board)
        # heuristic depth value -  a logarithmic function of # available actions
        n = len(avail_actions)
        if n == 6:
            d = 4
        elif n == 5:
            d = 5
        elif n == 4:
            d = 6
        elif n == 3:
            d = 8
        elif n == 2:
            d = 13
        else:
            d = 1
        '''Calculate action values using depth-limited heuristic-based minimax algorithm'''
        _, action_values = self.max_value_exp(board, depth=d)

        '''Select best action from available actions based on action values returned from minimax'''
        best_action = avail_actions[0]
        best_value = -float('inf')
        for i in avail_actions:
            if action_values[i] > best_value:
                best_action = i
                best_value = action_values[i]
        return best_action

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        """kernel_score rewards all vertical, horizontal, and diagonal winning situations"""
        r_score = self.kernel_score(board)
        c_score = self.kernel_score(board.T)
        return r_score + c_score

    def kernel_score(self, board):
        """creating kernels that consider all vertical, horizontal, and diagonal win situations"""
        """adding zeros in between rewards the diagonal winning situations"""
        max_kernels = ['{0}{0}'.format(1), '{0}{0}{0}'.format(1), '{0}{0}0{0}'.format(1), '{0}0{0}{0}'.format(1)]
        min_kernels = ['{0}{0}'.format(2), '{0}{0}{0}'.format(2), '{0}{0}0{0}'.format(2), '{0}0{0}{0}'.format(2)]
        weights = [1, 100, 100, 100]
        to_str = lambda a: ''.join(a.astype(str))

        max_kernel_count = 0
        min_kernel_count = 0
        for row in board:
            ''' Count the max player streaks'''
            for k, w in zip(max_kernels, weights):
                if k in to_str(row):
                    max_kernel_count += w

            ''' Count the min player streaks'''
            for k, w in zip(min_kernels, weights):
                if k in to_str(row):
                    min_kernel_count += w

        return max_kernel_count - min_kernel_count


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
