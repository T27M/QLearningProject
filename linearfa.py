
import numpy as np
import gym
import sys
import pickle
import os
import time
import json
import random


class LfaQAgent():
    def __init__(self):
        # Feature weights
        self.__w = 0.1 * np.ones((2, 4))
        self.__rewards = []

        self.__wrk_dir = './data/lfa/' + time.strftime("%Y%m%d-%H%M%S")

        if not os.path.exists(self.__wrk_dir):
            os.makedirs(self.__wrk_dir)

        self.__reward_path = self.__wrk_dir + '/lfa.reward.json'
        self.__weights_path = self.__wrk_dir + '/lfa.weights.pickle'

        self.__rewards = []

        # Learning rate
        self.alpha = 0.8

        self.alpha_zero = 0.8
        self.alpha_decay = 0.2

        # Discount factor
        self.gamma = 0.95

        # Random action
        self.epsilon = 0.1

        self.__actions = [0, 1]

    def paths(self):
        return self.__wrk_dir

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

    def weights(self):
        return self.__w

    def save_weights(self):
        with open(self.__weights_path, 'wb') as file:
            pickle.dump(self.__w, file)

    def load_weights(self, path):
        try:
            with open(path + 'lfa.weights.pickle', 'rb') as file:
                self.__w = pickle.load(file)
        except FileNotFoundError:
            print("Weight file not found")

    def Q(self, s, a):
        # Q value for state s
        return np.asscalar(np.dot(s, self.__w[a]))

    def Qs(self, s):
        return list(map(lambda a: self.Q(s, a), self.__actions))

    def maxQs(self, s):
        return np.max(self.Qs(s))

    def predict(self, s):
        if self.epsilon > random.random():
            return random.choice(self.__actions)
        else:
            return np.argmax(self.Qs(s))

    def update_fa(self, s, a, s1, r):
        Qsa = self.Q(s, a)
        maxQ = self.maxQs(s1)

        # Weight error
        error = (reward + self.gamma * maxQ) - Qsa

        # Multiply by error
        s = [self.alpha * error * x for x in s]

        # Update weights
        self.__w[a] = self.__w[a] + s

        # print(self.__w)

    def decay_learning_rate(self, episode):
        # 1/t decay
        self.alpha = self.alpha_zero / \
            (1 + self.alpha_decay * episode)


q = LfaQAgent()
# q.load_weights('./data/lfa/20180514-133021/')

print("Using weights..\n")
print(q.weights())

env = gym.make('CartPole-v0')
env.reset()

for episode in range(5000):
    print('\n')
    print('Episode: ' + str(episode))
    print("\tLearning Rate:" + str(q.alpha))

    episode_reward = 0
    action = 0
    s = env.reset()

    while(True):

        # env.render()

        # Predict action
        action = q.predict(s)

        # Take action
        s1, reward, done, info = env.step(action)

        # if done:
        # reward = -100

        # Update QTable
        q.update_fa(s, action, s1, reward)

        # print('\n')
        # print(q.weights())

        # Pass state along
        s = s1

        # Total episode reward
        episode_reward += reward

        if done:
            print("\tEpisode reward: " + str(episode_reward))
            q.add_reward(episode, episode_reward)

            print('\tSaving weights...')
            q.save_weights()

            # if q.alpha > 0.001:
            #     print('\tDecay learning rate')
            #     q.decay_learning_rate(episode)

            print('\n ---')
            break

print('\n Data saved to: ' + q.paths())

print('\n')
print(q.weights())
