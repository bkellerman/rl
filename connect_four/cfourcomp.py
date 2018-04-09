# Inspired by https://keon.io/deep-q-learning/

import numpy as np
from keras import backend as K
from cfourenv import ConnectFourGame
from cfouragent import DQNConnectFourAgent
from cfour import reverse_state
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8,
                                                   inter_op_parallelism_threads=8)))
EVERY_100 = lambda e: e % 100 == 0
EVERY_20 = lambda e: e % 20 == 0
RED = 'R'
YELLOW = 'Y'

class ConnectFourComp(object):
    def __init__(self, cols=7, rows=6, win=4, n_episodes=10000, save_interval=1000,
                 verbose=False):
        self.cols = cols
        self.rows = rows
        self.win = win
        self.n_episodes = n_episodes
        self.save_interval = save_interval
        self.verbose = verbose
        self.agent = DQNConnectFourAgent(cols=self.cols, rows=self.rows, win=self.win,
                                           alpha=0.001, alpha_decay=1.0, alpha_min=0.0001,
                                           epsilon_decay=0.99999, batch_size=256,
                                           online_update_freq=5, target_update_freq=25)

        self.env = ConnectFourGame(self.cols, self.rows, self.win)

    def preprocess_state(self):
        return np.zeros((1, self.cols * self.rows * 3))

    def run(self):

        for e in range(self.n_episodes):
            self.env.reset()
            done = False
            state = self.preprocess_state()

            # Red goes first
            c = RED
            while not done:

                # If yellow, reverse the state
                if c == YELLOW:
                    state = reverse_state(state, self.cols, self.rows)

                #state, action, reward, next_state, done = self.agent.turn(self.env, state, color=c)
                action = self.agent.turn(state)
                next_state, action, reward, done, _ = self.env.step(action, c)

                # Game is won
                if reward == 100:
                    # The opponent's last reward led to this loss, so decrease that reward
                    last = self.agent.memory.pop()
                    if last:
                        last_state, last_action, last_reward, last_next_state, last_done = last
                        last_reward = -100
                        last_done = True

                        if self.verbose:
                            board = state_to_board(last_state_sec, self.cols, self.rows)
                            print("last_state_sec")
                            print_board(board, self.cols, self.rows)
                            print("last_action_sec: %d, last_reward_sec: %0.2f, last_done_sec: %s" %
                                  (last_action_sec, last_reward_sec, last_done_sec))
                            next_board = state_to_board(last_next_state_sec, self.cols, self.rows)
                            print("last_next_state_sec")
                            print_board(next_board, self.cols, self.rows)

                        # Re-save opponent's last move
                        self.agent.remember(last_state, last_action, last_reward, last_next_state, last_done)

                    #done = True
                    #self.agent.remember(state, action, reward, next_state, done)
                    #continue

                # Game is won or tied
                #if done:
                    #self.agent.remember(state, action, reward, next_state, done)
                #    break
                    #continue

                # Play continues
                self.agent.remember(state, action, reward, next_state, done)

                # Game is won or tied
                if done:
                    #self.agent.remember(state, action, reward, next_state, done)
                    break

                # Move to next state
                state = next_state

                # Switch colors
                if c == RED:
                    c = YELLOW
                elif c == YELLOW:
                    c = RED

            self.agent.episode_over(e)

            if e % 100 == 0:
                print("e: %d, alpha: %0.4f, eps: %0.4f" %
                      (e, self.agent.alpha, self.agent.epsilon))
                self.env.game.print_board()

            if e % self.save_interval == 0:
                self.agent.save('agent.' + str(e))

if __name__ == '__main__':
    comp = ConnectFourComp(cols=7, rows=6, win=4, n_episodes=10000000)
    comp.run()
