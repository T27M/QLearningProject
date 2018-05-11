
import numpy as np
import gym
import sys
import pickle

actions = [0, 1]

# discount factor
gamma = 0.95

# decay rate
decay = 1

# learning rate
alpha = 0

alpha_zero = 0.2


class QAgent():
    def __init__(self):
        # Feature weights
        self.__w = 0.1 * np.ones((2, 4))

    def save(self):
        print('Saving weights...')

        with open('./data/lfa/lfa.pickle', 'wb') as file:
            pickle.dump(self.__w, file)

    def load(self):
        try:
            with open('./data/lfa/lfa.pickle', 'rb') as file:
                self.__w = pickle.load(file)

            print("Loaded weights..\n")
            print(self.__w)

        except FileNotFoundError:
            print("Weight file not found")

    def Q(self, s, a):
        # Q value for state s
        return np.asscalar(np.dot(s, self.__w[a]))

    def Qs(self, s):
        return list(map(lambda a: self.Q(s, a), actions))

    def maxQs(self, s):
        return np.max(self.Qs(s))

    def predict(self, s):
        return np.argmax(self.Qs(s))

    def update_fa(self, s, a, s1, r):
        Qsa = self.Q(s, a)
        maxQ = self.maxQs(s1)

        # Weight error
        error = reward + gamma * maxQ - Qsa

        # Multiply by error
        s = [(alpha * error) * x for x in s]

        # Update weights
        self.__w[a] = self.__w[a] + s

        # print(self.__w)


q = QAgent()

q.load()

env = gym.make('CartPole-v0')
env.reset()

rewards = []

for i in range(5000):

    print('Episode: ' + str(i))
    print("Learning Rate:" + str(alpha))

    episode_reward = 0
    action = 0
    s = env.reset()

    while(True):

        env.render()

        # Predict action
        action = q.predict(s)

        # Take action
        s1, reward, done, info = env.step(action)

        # Update QTable
        q.update_fa(s, action, s1, reward)

        # Pass state along
        s = s1

        # Total episode reward
        episode_reward += reward

        if done:

            print("Episode reward: " + str(episode_reward))
            rewards.append(episode_reward)

            with open("./data/lfa/reward.txt", "a") as myfile:
                myfile.writelines("Episode: " + str(i) +
                                  " | Reward: " + str(episode_reward) + "\n")

            q.save()

            print("Decaying learning rate..")

            alpha = 1 / (1 + decay * i) * alpha_zero

            print("Current learning rate: " + str(alpha))

            break
