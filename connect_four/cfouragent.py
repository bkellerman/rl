# Inspired by https://keon.io/deep-q-learning/
#import sys
import random
from collections import deque
#import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras import backend as K
#from cfourenv import ConnectFourGame
from cfour import state_to_board, print_board, reverse_state
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8,
                                                   inter_op_parallelism_threads=8)))
EVERY_100 = lambda e: e % 100 == 0
EVERY_20 = lambda e: e % 20 == 0

class DQNConnectFourAgent(object):
    def __init__(self, cols=7, rows=6, win=4, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, alpha=0.01, alpha_decay=0.9999, alpha_min=0.000001,
                 batch_size=64, online_update_freq=1, target_update_freq=5, verbose=False):

        self.memory = deque(maxlen=20000000)
        self.cols = cols
        self.rows = rows
        self.win = win
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.batch_size = batch_size
        self.online_update_freq = online_update_freq
        self.target_update_freq = target_update_freq
        self.verbose = verbose

        # Init model
        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def load_game(self, name, extension='.keras'):
        self.online_model.load_weights(name + extension)

    def load(self, name, extension='.keras'):
        self.online_model.load_weights('online.' + name + extension)
        self.target_model.load_weights('target.' + name + extension)

    def save(self, name, extension='.keras'):
        self.online_model.save_weights('online.' + name + extension)
        self.target_model.save_weights('target.' + name + extension)

    """
    def _build_model(self):
        self.opt = Adam(lr=self.alpha)
        model = Sequential()
        model.add(Reshape((6, 7, 3), input_shape=(self.cols * self.rows * 3,)))
        model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(32, (2, 2), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(64, (2, 2), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        model.add(Dense(self.cols, activation='linear'))
        model.compile(loss='mse', optimizer=self.opt)
        return model

    """
    def _build_model(self):
        self.opt = Adam(lr=self.alpha)
        model = Sequential()
        model.add(Dense(128, input_shape=(self.cols * self.rows * 3,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.cols, activation='linear'))
        model.compile(loss='mse', optimizer=self.opt)
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.online_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_online_action(self, state):
        if np.random.random() <= self.epsilon:
            return np.array(random.randint(0, self.cols - 1))

        return np.argmax(self.online_model.predict(state)[0])

    def choose_game_action(self, state):
        return np.argmax(self.online_model.predict(state)[0])

    def preprocess_state(self):
        s = np.zeros((1, self.cols * self.rows * 3))
        return s

    def replay(self, batch_size, verbose=False):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            y_target = self.online_model.predict(state)
            if done:
                y_target[0][action] = reward
            else:
                a = self.online_model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                y_target[0][action] = reward + self.gamma * t[np.argmax(a)]

            if self.verbose:
                print_board(state_to_board(state, self.cols, self.rows), self.cols, self.rows)
                print(y_target[0])

            x_batch.append(state[0])
            y_batch.append(y_target[0])

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        self.online_model.fit(x_batch, y_batch, batch_size=len(x_batch), epochs=2, verbose=verbose)

    def turn_old(self, env, state, color):
        # Determine action
        action = self.choose_online_action(state)

        # Perform action for this color
        next_state, reward, done, info = env.turn(action, color)

        return state, action, reward, next_state, done

    def turn(self, state):
        # Determine action
        action = self.choose_online_action(state)

        return action

        # Perform action for this color
        #next_state, reward, done, info = env.turn(action, color)

        #return state, action, reward, next_state, done

    def episode_over(self, e):
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Decay learning rates
        if self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay

        K.set_value(self.online_model.optimizer.lr, self.alpha)
        K.set_value(self.target_model.optimizer.lr, self.alpha)

        if e % self.online_update_freq == 0:
            if e % 100 == 0:
                self.replay(self.batch_size, verbose=True)
            else:
                self.replay(self.batch_size)

        if e % self.target_update_freq == 0:
            self.update_target_model()
