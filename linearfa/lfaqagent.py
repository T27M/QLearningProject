import sys
import time
import os
import numpy as np
import json
import pickle
import random
from pathlib import Path
import matplotlib.pyplot as plt


class LfaQAgent(object):
    def __init__(self, learning_rate, discount_factor, environment):

        self.__env_pacman = 'MsPacman-v0'
        self.__env_cartpole = 'CartPole-v0'

        self.__environment = environment

        if self.__environment == self.__env_pacman:
            # Pacman Feature weights
            self.__w = 0.1 * np.ones((5, 163), dtype=np.float64)
        else:
            # CartPole Feature Weights
            self.__w = np.random.sample(size=(2, 4))
            print(self.__w)

        self.__rewards = []

        self.__gen_dir()

        # call(['gnome-terminal', '-x', 'tail -f '])

        self.__rewards = []
        self.__error = []

        # Learning rate
        self.alpha = learning_rate
        print(self.alpha)
        self.__min_learning_rate = 0.0001

        self.alpha_zero = 0.8
        self.alpha_decay = 0.5

        # Discount factor
        self.gamma = discount_factor

        # Random action
        self.epsilon = 0.1

        if self.__environment == self.__env_pacman:
            # Pacman
            self.__actions = [1, 2, 3, 4]
        else:
            # CartPole
            self.__actions = [0, 1]

        self.__step_decay = 250
        self.__decay_step_ctr = 0

    def __gen_dir(self):
        self.__wrk_dir = './data/lfa/' + time.strftime("%Y%m%d-%H%M%S")

        if not os.path.exists(self.__wrk_dir):
            os.makedirs(self.__wrk_dir)

        self.__reward_path = self.__wrk_dir + '/lfa.reward.json'
        self.__weights_path = self.__wrk_dir + '/{}/lfa.weights.pickle'
        self.__error_path = self.__wrk_dir + '/lfa.error.json'

    def stats(self):
        print('\n Weights:')
        print(self.weights())

        print('\n Best episode:')
        print(self.best_episode())

        print('\n Avg score/episode:')
        print(str(self.avg_score()))

        print('\n Data saved to: ' + self.paths())

    def paths(self):
        return self.__wrk_dir

    def best_episode(self):
        best = max(self.__rewards, key=lambda x: x['episode_reward'])
        return best

    def avg_score(self):
        episodes = len(self.__rewards)
        scores = [x['episode_reward'] for x in self.__rewards]
        total = sum(scores)

        return total / episodes

    def clear_score(self):
        self.__gen_dir()
        self.__rewards = []

    def add_reward(self, episode, episode_reward):
        er_dict = {
            'episode': episode,
            'episode_reward': episode_reward
        }

        try:
            with open(self.__reward_path, 'r') as file:
                self.__rewards = json.load(file)
        except:
            pass

        self.__rewards.append(er_dict)

        with open(self.__reward_path, 'w') as file:
            json.dump(self.__rewards, file)

    def add_error(self, error):
        er_dict = {
            'error': error,
        }

        self.__error.append(er_dict)

    def save_error(self):
        with open(self.__error_path, 'w') as file:
            json.dump(self.__error, file)

    def weights(self):
        return self.__w

    def save_weights(self, episode):
        if np.isnan(self.__w).any():
            print('Skipping save - Nan detected')
        else:
            if not os.path.exists(self.__wrk_dir + '/' + str(episode)):
                os.makedirs(self.__wrk_dir + '/' + str(episode))

            weight_episode_path = self.__weights_path.format(episode)

            with open(weight_episode_path, 'wb') as file:
                pickle.dump(self.__w, file)

    def load_weights(self, path):
        try:
            with open(path + 'lfa.weights.pickle', 'rb') as file:
                self.__w = pickle.load(file)
                print('Loaded weights!')
                print(self.__w)
                input('\n Press any key to continue...')

        except FileNotFoundError:
            print("Weight file not found")

    def get_Q(self, s, a):
        # Q value for state s
        return np.dot(s, self.__w[a])

    def get_all_Q(self, s):
        return list(map(lambda a: self.get_Q(s, a), self.__actions))

    def get_max_Q(self, s):
        return np.max(self.get_all_Q(s))

    def predict(self, s):
        if self.epsilon > random.random():
            return random.choice(self.__actions)
        else:
            return self.__actions[np.argmax(self.get_all_Q(s))]

    def act(self, s):
        return self.__actions[np.argmax(self.get_all_Q(s))]

    def update_fa(self, s, a, s1, r):

        # print(s)
        # print(a)
        # print(s1)
        # print(r)
        # input('')

        # if(np.array_equal(s, s1)):
        #     print("ERROR: s == s1")
        #     sys.exit()

        action_index = a
        # print("AI:" + str(action_index))
        Qsa = self.get_Q(s, a)
        maxQ = self.get_max_Q(s1)

        # Q-Learning
        q_td = r + self.gamma * maxQ

        difference = q_td - Qsa

        self.add_error(difference)

        weight_update = [self.alpha * difference * fi for fi in s]

        # Update weights
        for weight_i in range(len(self.__w[action_index])):
            cur_weight = self.__w[action_index][weight_i]
            new_weight = cur_weight + weight_update[weight_i]

            self.__w[action_index][weight_i] = new_weight

    # def update_fa(self, s, a, s1, r):
    #     Qsa = self.get_Q(s, a)
    #     maxQ = self.get_max_Q(s1)

    #     if(np.array_equal(s, s1)):
    #         print("ERROR: s == s1")
    #         sys.exit()

    #     # Weight error
    #     td_delta = (r + self.gamma * maxQ) - Qsa

    #     print(td_delta)

    #     # Multiply by error
    #     ms = [self.alpha * td_delta * x for x in s]

    #     print(self.__w)
    #     print(ms)

    #     # Update weights
    #     self.__w[a] = self.__w[a] + ms

    #     print(self.__w)

    #     input('')

        # Round weights
        # self.__w[a] = [np.around(x, 3) for x in self.__w[a]]

    # def update_fa(self, s, a, s1, r):
    #     Qsa = self.get_Q(s, a)
    #     maxQ = self.get_max_Q(s1)

    #     if(np.array_equal(s, s1)):
    #         print("ERROR: s == s1")
    #         sys.exit()

    #     # print("QSA " + str(Qsa))
    #     # print("MaxQ " + str(maxQ))

    #     # Weight error
    #     td_delta = r + (self.gamma * maxQ) - Qsa

    #     print(td_delta)

    #     # Multiply by error
    #     ms = [self.alpha * td_delta * x for x in s]

    #     # Update weights
    #     self.__w[a] = self.__w[a] + ms

    #     # Round weights
    #     # self.__w[a] = [np.around(x, 3) for x in self.__w[a]]

    def decay_learning_rate(self, episode):
        if self.__decay_step_ctr == self.__step_decay:
            self.__decay_step_ctr = 0
            self.alpha -= self.alpha * self.alpha_decay
            return True
        else:
            self.__decay_step_ctr = self.__decay_step_ctr + 1
            return False

            # # 1/t decay
            # self.alpha = self.alpha_zero / \
            #     (1 + self.alpha_decay * episode)
