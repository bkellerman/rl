# Inspired by https://keon.io/deep-q-learning/

import random
from collections import deque
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8,
                                                   inter_op_parallelism_threads=8)))
EVERY_100 = lambda e: e % 100 == 0

class DQNMountainCarSolver():
    def __init__(self, n_episodes=1000, gamma=0.9999, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, alpha=0.1, alpha_decay=0.99995, alpha_min=0.000001,
                 batch_size=256, online_update_freq=1, target_update_freq=5, quiet=False):

        self.memory = deque(maxlen=10000000)
        env = gym.make("MountainCarContinuous-v0")
        self.max_steps = 2000
        env._max_episode_steps = self.max_steps
        #self.env = gym.wrappers.Monitor(self.env, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) +
        #                               '/lunarlander_monitor_nopretrain_eps995_lr001', video_callable=every100, force=True)

        self.env = gym.wrappers.Monitor(env, 'mountaincar', video_callable=EVERY_100, force=True)

        self.n_episodes = n_episodes
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
        self.quiet = quiet

        # Init model
        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        self.opt = Adam(lr=self.alpha)
        model = Sequential()
        model.add(Dense(128, input_shape=(2,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=self.opt)
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.online_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_online_action(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()

        return self.online_model.predict(state)[0]

    def preprocess_state(self, state):
        s = np.zeros((1, 2))
        return s

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            y_target = self.online_model.predict(state)
            if done:
                #y_target[0] = reward
                y_target[0] = action[0]
            else:
                a = self.online_model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                #y_target[0][action] = reward + self.gamma * t[np.argmax(a)]
                y_target[0] = reward + self.gamma * t[0]

            x_batch.append(state[0])
            y_batch.append(y_target[0])

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        self.online_model.fit(x_batch, y_batch, batch_size=len(x_batch), epochs=1, verbose=0)

    def target_replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))

        update_online = np.random.randint(0, 2)
        if update_online:
            model_to_update = self.online_model
        else:
            model_to_update = self.target_model

        for state, action, reward, next_state, done in minibatch:
            if update_online:
                y_target = self.online_model.predict(state)
            else:
                y_target = self.target_model.predict(state)

            if done:
                y_target[0][action] = reward
            else:
                if update_online:
                    a = self.online_model.predict(next_state)[0]
                    t = self.target_model.predict(next_state)[0]
                    y_target[0][action] = reward + self.gamma * t[np.argmax(a)]
                else:
                    a = self.target_model.predict(next_state)[0]
                    t = self.online_model.predict(next_state)[0]
                    y_target[0][action] = reward + self.gamma * t[np.argmax(a)]

            x_batch.append(state[0])
            y_batch.append(y_target[0])

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        model_to_update.fit(x_batch, y_batch, batch_size=len(x_batch), epochs=2, verbose=0)

    def run(self):
        rewards = deque(maxlen=100)
        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            total_reward = 0
            i = 1
            while not done:
                action = self.choose_online_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, 2])
                #if done and i < self.max_steps:
                #    reward = 1
                    #for _ in range(1):
                    #    self.remember(state, action, reward, next_state, done)
                self.remember(state, action, reward, next_state, done)

                state = next_state
                i += 1

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if self.alpha > self.alpha_min:
                self.alpha *= self.alpha_decay

            # Decay learning rates
            K.set_value(self.online_model.optimizer.lr, self.alpha)
            K.set_value(self.target_model.optimizer.lr, self.alpha)

            rewards.append(total_reward)
            mean_reward = np.mean(rewards)

            if e % 100 == 0:
                print("e: %d, lr: %0.5f, mean total reward: %d" %
                      (e, K.get_value(self.target_model.optimizer.lr), mean_reward))

            if mean_reward >= 200:
                print("Solved on episode %d. Mean reward: %0.2f" % (e, mean_reward))
                break

            if e % self.online_update_freq == 0:
                self.replay(self.batch_size)

            if e % self.target_update_freq == 0:
                self.update_target_model()

if __name__ == '__main__':
    agent = DQNMountainCarSolver(n_episodes=10000000)
    agent.run()
