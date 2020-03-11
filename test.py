import numpy as np

""" 
Player 1: max player
Player 2: min player
"""


def update_board(board, move, player_num):
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
        return 1
    elif game_completed(board, player_num=2):
        return -1
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
        # print('------max-------')
        # print(state, alpha, beta)
        action_values = [0 for _ in range(state.shape[1])]  # -2 is correct?????
        utility = terminal_state(state)
        if utility is not None:  # Game has a winner
            return utility, action_values
        avail_actions = available_actions(state)
        if len(avail_actions) == 0:  # game is tie
            return 0, action_values

        if depth == 0:
            return evaluation_function(state), action_values

        v = -float('inf')
        for a in avail_actions:
            state_ = update_board(state.copy(), a, player_num=1)  # next state
            v = max(v, self.min_value(state_, alpha, beta, depth - 1))
            action_values[a] = v
            if v >= beta: return v, action_values
            alpha = max(alpha, v)
        return v, action_values

    def min_value(self, state, alpha, beta, depth):
        # print('------min-------')
        # print(state, alpha, beta)
        utility = terminal_state(state)
        if utility is not None:  # Game has a winner
            return utility
        avail_actions = available_actions(state)
        if len(avail_actions) == 0:  # game is tie
            return 0

        if depth == 0:
            return evaluation_function(state)

        v = +float('inf')
        for a in avail_actions:
            state_ = update_board(state.copy(), a, player_num=2)  # next state
            mxv, _ = self.max_value(state_, alpha, beta, depth - 1)
            v = min(v, mxv)
            if v <= alpha: return v
            beta = min(beta, v)
        return v

    def evaluation_function(self, board):
        win_kernel2 = '{0}{0}'.format(1)
        win_kernel3 = '{0}{0}{0}'.format(1)
        lose_kernel2 = '{0}{0}'.format(2)
        lose_kernel3 = '{0}{0}{0}'.format(2)
        win_two_count = 0
        win_three_count = 0
        lose_two_count = 0
        lose_three_count = 0
        to_str = lambda a: ''.join(a.astype(str))

        for row in board:
            ''' Count the max player streaks'''
            if win_kernel2 in to_str(row):
                win_two_count += 1
            if win_kernel3 in to_str(row):
                win_three_count += 1

            ''' Count the min player streaks'''
            if lose_kernel2 in to_str(row):
                lose_two_count += 1
            if lose_kernel3 in to_str(row):
                lose_three_count += 1
        return 0

def row_score(board):
    max_kernels = ['{0}{0}'.format(1), '{0}{0}{0}'.format(1), '{0}{0}0{0}'.format(1), '{0}0{0}{0}'.format(1)]
    min_kernels = ['{0}{0}'.format(2), '{0}{0}{0}'.format(2), '{0}{0}0{0}'.format(2), '{0}0{0}{0}'.format(2)]
    weights = [2, 10, 10, 10]
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

def evaluation_function(board):
    r_score = row_score(board)
    c_score = row_score(board.T)
    return int((r_score + c_score) / 2)

# player = AIPlayer(1)
# board = np.zeros([6, 7]).astype(np.uint8)
# player.get_alpha_beta_move(board)

board = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

board = np.array([[0, 0, 0, 0],
                  [0, 2, 0, 0],
                  [2, 2, 0, 0],
                  [1, 0, 1, 1]])

score = evaluation_function(board)
v= [1,2,3,4]
print()