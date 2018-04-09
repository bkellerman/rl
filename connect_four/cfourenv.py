import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from cfour import Game

class ConnectFourGame(gym.Env):
    def __init__(self, cols=7, rows=6, win=4):
        self.cols = cols
        self.rows = rows
        self.win = win
        self.game = Game(self.cols, self.rows, self.win)
        self.opponent = 'one_block'
        self.opponent = 'all_block'

        self.action_space = spaces.Box(low=np.array([0]), high=np.array([self.cols]))
        self.observation_space = spaces.Discrete(self.cols * self.rows * 2)

        self.state = np.zeros((self.cols, self.rows, 3))

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def valid_step(self, action):
        return self.game.open_col(action)

    def step_old(self, action):
        #assert self.action_space.contains(action)

        done = False
        reward = 0

        self.game.insert(action, 'R')
        if self.game.check_for_win():
            reward = 100
            done = True
            #self.game.print_board()
        else:
            if self.opponent == 'random':
                self.game.insert_random('Y')
            elif self.opponent == 'one_block':
                self.one_block('Y')
            elif self.opponent == 'all_block':
                self.all_block('Y')
            if self.game.check_for_win():
                reward = -100
                done = True

        if self.game.full():
            done = True

        return self.game.state(), reward, done, {}

    def step(self, action, color):

        done = False
        reward = 0

        if not self.game.insert(action, color):
            reward = -100

        if self.game.check_for_win():
            reward = 100
            done = True

        if self.game.full():
            done = True

        return self.game.state(), action, reward, done, {}

    def reset(self):
        self.game = Game(self.cols, self.rows, self.win)
        self.state = np.zeros((self.cols, self.rows, 3))
        return self.state
