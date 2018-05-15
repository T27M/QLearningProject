
import numpy as np
import gym
import sys
import pickle
import os
import time
import json
import random

from core.featureprocessor import FeatureProcessor
from core.config import Config


class LfaQAgent():
    def __init__(self):
        # Feature weights
        self.__w = 0.1 * np.ones((4, 163), dtype=np.float64)
        self.__rewards = []

        self.__wrk_dir = './data/lfa/' + time.strftime("%Y%m%d-%H%M%S")

        if not os.path.exists(self.__wrk_dir):
            os.makedirs(self.__wrk_dir)

        self.__reward_path = self.__wrk_dir + '/lfa.reward.json'
        self.__weights_path = self.__wrk_dir + '/lfa.weights.pickle'

        self.__rewards = []

        # Learning rate
        self.alpha = 0.2
        self.__min_learning_rate = 0.01

        self.alpha_zero = 0.8
        self.alpha_decay = 0.05

        # Discount factor
        self.gamma = 0.9

        # Random action
        self.epsilon = 0.1

        self.__actions = [1, 2, 3, 4]

        self.__step_decay = 100
        self.__decay_step_ctr = 0

    def stats(self):
        print('\n Data saved to: ' + self.paths())

        print('\n Weights:')
        print(self.weights())

        print('\n Best episode:')
        print(self.best_episode())

        print('\n Avg score/episode:')
        print(str(self.avg_score()))

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
        return np.asscalar(np.dot(s, self.__w[a - 1]))

    def Qs(self, s):
        return list(map(lambda a: self.Q(s, a), self.__actions))

    def maxQs(self, s):
        return np.max(self.Qs(s))

    def predict(self, s):
        if self.epsilon > random.random():
            return random.choice(self.__actions)
        else:
            return self.__actions[np.argmax(self.Qs(s))]

    def act(self, s):
        return np.argmax(self.Qs(s))

    def update_fa(self, s, a, s1, r):
        Qsa = self.Q(s, a)
        maxQ = self.maxQs(s1)

        # Weight error
        error = (r + self.gamma * maxQ) - Qsa

        # Multiply by error
        ms = [self.alpha * error * x for x in s]

        # Update weights
        self.__w[a - 1] = self.__w[a - 1] + ms

        # Round weights
        self.__w[a - 1] = [np.around(x, 3) for x in self.__w[a - 1]]

        # print(self.__w)

    def decay_learning_rate(self, episode):
        if self.alpha - self.alpha_decay >= self.__min_learning_rate:
            if self.__decay_step_ctr == self.__step_decay:
                self.__decay_step_ctr = 0
                self.alpha = self.alpha - self.alpha_decay
                return True
            else:
                self.__decay_step_ctr = self.__decay_step_ctr + 1
                return False

                # # 1/t decay
                # self.alpha = self.alpha_zero / \
                #     (1 + self.alpha_decay * episode)
        else:
            self.alpha = self.__min_learning_rate


class Env():
    def __init__(self, agent, render, train, decay):
        self.__render = render
        self.__train = train
        self.__decay = decay
        self.__agent = agent

        self.__starting_lives = 3
        self.__current_lives = 3

        if train:
            print("\tTraining agent...")
        else:
            print("\tUsing policy...")

        print('\tUsing weights..\n')
        print(q.weights())

    def run(self, episodes):
        # env = gym.make('CartPole-v0')
        env = gym.make('MsPacman-v0')
        env.reset()

        config = Config()
        fp = FeatureProcessor(config)

        for episode in range(episodes):
            print('\n')
            print('Episode: ' + str(episode))
            print("\tLearning Rate:" + str(q.alpha))

            episode_reward = 0
            action = 1
            s = env.reset()

            s = fp.extract_features(s)

            while(True):
                if self.__render:
                    env.render()

                if train:
                    # Predict action
                    action = self.__agent.predict(s)
                else:
                    # Use policy
                    action = self.__agent.act(s)

                # Take action
                s1, reward, done, info = env.step(action)
                s1 = fp.extract_features(s1)

                if info['ale.lives'] < self.__current_lives:
                    self.__current_lives = self.__current_lives - 1
                    print('Lost life')
                    reward = -1000

                # if done:
                # reward = -1000

                if train:
                    # Update QTable
                    self.__agent.update_fa(s, action, s1, reward)

                # print('\n')
                # print(q.weights())

                # Pass state along
                s = s1

                # Total episode reward
                episode_reward += reward

                if done:
                    self.__current_lives = self.__starting_lives

                    print("\tEpisode reward: " + str(episode_reward))
                    self.__agent.add_reward(episode, episode_reward)

                    if self.__train:
                        print('\tSaving weights...')
                        self.__agent.save_weights()

                        if self.__decay:
                            if self.__agent.decay_learning_rate(episode):
                                print('\tDecay learning rate..')

                    print('\n ---')
                    break


render = False
train = True
decay = False
episodes = 100

q = LfaQAgent()
# q.load_weights('./data/lfa/20180515-112554/')

en = Env(q, render, train, decay)
en.run(episodes)

q.stats()

while True:
    key = ""
    while key.lower() not in ['r', 'rr', 't', 'q']:
        key = input(
            '(r: run the policy)(rr:run the policy and render(NOTE: this is very slow!))(t: train again with same settings)(q: quit):')

    if key.lower() == "r":
        train = False

    if key.lower() == 'rr':
        train = False
        render = True

    if key.lower() == 'q':
        break

    q.clear_score()
    en = Env(q, render, train, decay)
    en.run(episodes)
    q.stats()
