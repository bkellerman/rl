# https://gist.github.com/poke
from itertools import groupby, chain
import random
import numpy as np

NONE = '.'
RED = 'R'
YELLOW = 'Y'

def diagonals_pos(matrix, cols, rows):
    """Get positive diagonals, going from bottom-left to top-right."""
    for di in ([(j, i - j) for j in range(cols)] for i in range(cols + rows -1)):
        yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

def diagonals_neg(matrix, cols, rows):
    """Get negative diagonals, going from top-left to bottom-right."""
    for di in ([(j, i - cols + j + 1) for j in range(cols)] for i in range(cols + rows - 1)):
        yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

def reverse_state(state, cols, rows):
    arrs = np.split(state[0], cols * rows)
    for a in arrs:
        if a[2] == 1:
            continue
        elif a[0] == 1:
            a[0] = 0
            a[1] = 1
        else:
            a[0] = 1
            a[1] = 0
    return np.concatenate(arrs).reshape(1, -1)

def state_to_board(state, cols, rows):
    board = [[NONE] * rows for _ in range(cols)]
    arrs = np.split(state[0], cols * rows)
    for i in range(cols):
        for j in range(rows):
            a = arrs[i*rows + j]
            if a[0] == 1:
                board[i][j] = RED
            elif a[1] == 1:
                board[i][j] = YELLOW
            else:
                board[i][j] = NONE
    return board

def print_board(board, cols, rows):
    """Print the board."""
    print('  '.join(map(str, range(cols))))
    for y in range(rows):
        print('  '.join(str(board[x][y]) for x in range(cols)))
    print()

class Game:
    def __init__(self, cols=5, rows=4, win=3):
        """Create a new game."""
        self.cols = cols
        self.rows = rows
        self.win = win
        self.board = [[NONE] * rows for _ in range(cols)]

    def active_cols(self):
        cols = []
        for x in range(self.cols):
            c = self.board[x]
            if c[-1] != NONE:
                cols.append(x)

        return cols

    def top_color_cols(self, color):
        cols = []
        for x in range(self.cols):
            if self.top_color(x) == color:
                cols.append(x)
        return cols

    def top_color(self, column):
        c = self.board[column]
        i = -1
        while i >= -1 * self.rows:
            if c[i] != NONE:
                return c[i]
            i -= 1

        return NONE

    def open_col(self, col):
        c = self.board[col]
        return c[0] == NONE

    def insert(self, column, color):
        """Insert the color in the given column."""
        c = self.board[column]
        if c[0] != NONE:
            return False

        i = -1
        while c[i] != NONE:
            i -= 1
        c[i] = color

        return True

    def insert_random(self, color):
        cols = list(range(self.cols))
        random.shuffle(cols)

        for n in cols:
            c = self.board[n]
            if c[0] != NONE:
                continue

            i = -1
            while c[i] != NONE:
                i -= 1
            c[i] = color
            break

    def col_full(self, column):
        c = self.board[column]
        return c[0] != NONE

    def full(self):
        for i in range(self.cols):
            c = self.board[i]
            if c[0] == NONE:
                return False

        return True

    def check_for_win(self):
        """Check the current board for a winner."""
        return bool(self.get_winner())

    def get_winner(self):
        """Get the winner on the current board."""
        lines = (self.board, # columns
                 zip(*self.board), # rows
                 diagonals_pos(self.board, self.cols, self.rows), # positive diagonals
                 diagonals_neg(self.board, self.cols, self.rows)) # negative diagonals

        for line in chain(*lines):
            for color, group in groupby(line):
                if color != NONE and len(list(group)) >= self.win:
                    return color

    def state(self):
        state = np.zeros((self.cols, self.rows, 3))
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[x][y] == RED:
                    state[x, y, 0] = 1
                elif self.board[x][y] == YELLOW:
                    state[x, y, 1] = 1
                else:
                    state[x, y, 2] = 1

        return state.flatten().reshape(1, -1)

    def reversed_state(self):
        state = np.zeros((self.cols, self.rows, 3))
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[x][y] == RED:
                    state[x, y, 1] = 1
                elif self.board[x][y] == YELLOW:
                    state[x, y, 0] = 1
                else:
                    state[x, y, 2] = 1

        return state.flatten().reshape(1, -1)
    def state_to_board(self, state):
        board = [[NONE] * self.rows for _ in range(self.cols)]
        arrs = np.split(state[0], self.cols * self.rows)
        for i in range(self.cols):
            for j in range(self.rows):
                a = arrs[i*self.rows + j]
                if a[0] == 1:
                    board[i][j] = RED
                elif a[1] == 1:
                    board[i][j] = YELLOW
                else:
                    board[i][j] = NONE
        return board

    def n_tokens(self):
        n = 0
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[x][y] != NONE:
                    n += 1
        return n

    def print_board(self, board=None):
        """Print the board."""
        if board is None:
            board = self.board

        print('  '.join(map(str, range(self.cols))))
        for y in range(self.rows):
            print('  '.join(str(board[x][y]) for x in range(self.cols)))
        print()
