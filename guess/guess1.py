# Inspired by https://keon.io/deep-q-learning/

import random
import copy
import gym
from gym.envs.toy_text import GuessingGame
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras import backend as K
import tensorflow as tf
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6)))
class DQNGuessingGameSolver():
    def __init__(self, n_episodes=1000, n_win_score=4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, alpha=0.01,
                 alpha_decay=1.0, alpha_min=0.000001, batch_size=256, pre_train_steps=0,
                 online_update_freq=1, target_update_freq=5, quiet=False):
        self.memory = deque(maxlen=100000)
        self.pos_memory = deque(maxlen=100000)
        self.env = GuessingGame()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.n_episodes = n_episodes
        self.n_win_score = n_win_score
        self.batch_size = batch_size
        self.pre_train_steps = pre_train_steps
        self.online_update_freq = online_update_freq
        self.target_update_freq = target_update_freq
        self.quiet = quiet

        # Init model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(256, input_shape=(30, 13), activation='tanh', return_sequences=True))
        model.add(LSTM(128, activation='tanh', return_sequences=True))
        model.add(LSTM(128, activation='tanh', return_sequences=False))
        #model.add(Dense(100, activation='linear'))
        #model.add(Dense(30, activation='softmax'))
        model.add(Dense(1024, activation='linear'))
        #model.add(Dense(1024, activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        #model.compile(loss=self._huber_loss, optimizer=Adam(clipnorm=1., lr=self.alpha, decay=self.alpha_decay))
        #model.compile(loss='mse', optimizer=Adam(clipnorm=1., lr=self.alpha, decay=self.alpha_decay))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        #print("State: %s, action: %s, reward: %s, next_state: %s, done: %s"%(str(state), str(action), str(reward), str(next_state), str(done)))
        #print("action: %s, reward: %s, done: %s"% (str(action), str(reward), str(done)))
        self.memory.append((state, action, reward, next_state, done))

    def pos_remember(self, state, action, reward, next_state, done):
        #print("State: %s, action: %s, reward: %s, next_state: %s, done: %s"%(str(state), str(action), str(reward), str(next_state), str(done)))
        #print("action: %s, reward: %s, done: %s"% (str(action), str(reward), str(done)))
        self.pos_memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
        #if np.random.random() <= self.epsilon:
            #print("choose_action: random")
            #action = np.array(np.random.randint(1, 31)).reshape(1,)
            action = np.array(np.random.randint(1, 1025)).reshape(1,)
            return action

        #print("choose_action: predict")
        #print(np.argmax(self.model.predict(state)))
        return np.argmax(self.model.predict(state)).reshape(1,) + 1

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        s = np.zeros((1, 30, 13))
        #s = np.zeros((1, 10, 8))
        return s

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))

        pos_minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        batch = minibatch + pos_minibatch
        for state, action, reward, next_state, done in batch:
            # add noise
            #action = min(max(action + np.random.randint(-10, 10), 1), 1024)
            y_target = self.model.predict(state)
            if done:
                y_target[0][action - 1] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                y_target[0][action - 1] = reward + self.gamma * t[np.argmax(a)]

            x_batch.append(state[0])
            y_batch.append(y_target[0])

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        self.model.fit(x_batch, y_batch, batch_size=len(x_batch), epochs=1, verbose=0, shuffle=True)


    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state)
                ns, reward, done, d = self.env.step(action)
                #if e % 100 == 0 and e > self.pre_train_steps:
                #    print("ns: %d, action: %d, done: %s, d: %s" % (ns, action, done, d))

                #next_state = copy.copy(state)
                next_state = state[:]
                """
                next_state[0, :, 0] = np.append(next_state[0, :, 0], ns / 3)[1:]
                """
                if ns == 1:
                    next_state[0, :, 0] = np.append(next_state[0, :, 0], 1)[1:]
                    next_state[0, :, 1] = np.append(next_state[0, :, 1], -1)[1:]
                    next_state[0, :, 2] = np.append(next_state[0, :, 2], -1)[1:]
                elif ns == 2:
                    next_state[0, :, 0] = np.append(next_state[0, :, 0], -1)[1:]
                    next_state[0, :, 1] = np.append(next_state[0, :, 1], 1)[1:]
                    next_state[0, :, 2] = np.append(next_state[0, :, 2], -1)[1:]
                elif ns == 3:
                    next_state[0, :, 0] = np.append(next_state[0, :, 0], -1)[1:]
                    next_state[0, :, 1] = np.append(next_state[0, :, 1], -1)[1:]
                    next_state[0, :, 2] = np.append(next_state[0, :, 2], 1)[1:]

                """
                next_state[0, :, 1] = np.append(next_state[0, :, 1], action / 30)[1:]
                """
                action_bin = bin(action[0])[2:].zfill(10)
                #action_bin = bin(action[0])[2:].zfill(5)
                action_bin = list(action_bin)
                for index, item in enumerate(action_bin):
                    if item ==  '0':
                        action_bin[index] = '-1'
                """
                # Replace 0,1
                action_bin = list(action_bin)
                for index, item in enumerate(action_bin):
                    if item == '0':
                        action_bin[index] = '0.333'
                    elif item == '1':
                        action_bin[index] = '0.666'

                """
                next_state[0, :, 3] = np.append(next_state[0, :, 3], float(action_bin[0]))[1:]
                next_state[0, :, 4] = np.append(next_state[0, :, 4], float(action_bin[1]))[1:]
                next_state[0, :, 5] = np.append(next_state[0, :, 5], float(action_bin[2]))[1:]
                next_state[0, :, 6] = np.append(next_state[0, :, 6], float(action_bin[3]))[1:]
                next_state[0, :, 7] = np.append(next_state[0, :, 7], float(action_bin[4]))[1:]
                next_state[0, :, 8] = np.append(next_state[0, :, 8], float(action_bin[5]))[1:]
                next_state[0, :, 9] = np.append(next_state[0, :, 9], float(action_bin[6]))[1:]
                next_state[0, :, 10] = np.append(next_state[0, :, 10], float(action_bin[7]))[1:]
                next_state[0, :, 11] = np.append(next_state[0, :, 11], float(action_bin[8]))[1:]
                next_state[0, :, 12] = np.append(next_state[0, :, 12], float(action_bin[9]))[1:]

                if reward == 1:
                    for _ in range(100):
                        self.pos_remember(state, action, reward, next_state, done)
                else:
                    self.remember(state, action, reward, next_state, done)

                state = next_state
                i += 1

            if e < self.pre_train_steps:
                continue

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if self.alpha > self.alpha_min:
                self.alpha *= self.alpha_decay

            K.set_value(self.model.optimizer.lr, self.alpha)
            K.set_value(self.target_model.optimizer.lr, self.alpha)

            scores.append(i)
            mean_score = np.mean(scores)
            #print("e: %d, score: %d, mean_score: %0.2f" % (e, i, mean_score))
            if mean_score <= self.n_win_score and e >= 100:
                if not self.quiet: print('Ran {} episodes. Mean {}. Solved after {} trials ✔'.format(e, mean_score, e))
                return e

            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean score over last 100 episodes was {} ticks.'.format(e, mean_score))

            if e % self.online_update_freq == 0:
                self.replay(self.batch_size)

            if e % self.target_update_freq == 0:
                self.update_target_model()

        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNGuessingGameSolver(n_episodes=20000000, n_win_score=10)
    #agent = DQNGuessingGameSolver(n_episodes=10000000, n_win_score=5)
    agent.run()
