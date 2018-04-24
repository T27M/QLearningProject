import gym
from gym import wrappers
import numpy as np
from core.config import Config
import time


class Environment(object):
    def __init__(self, config: Config, feature_processor, agent):
        self.__debug = config.get_config('debug')
        self.__log_output = config.get_config('log_output')
        self.__timestr = time.strftime("%Y%m%d-%H%M%S")
        self.__log_path = config.get_config('log_path') + \
            type(self).__name__ + self.__timestr

        self.__config = config.get_config(type(self).__name__)

        self.__cumulative_reward = 0
        self.__reward = 0
        self.__reward_list = []

        # Render the game enviroment
        self.__render = self.__config['render']

        # Number of episodes
        self.__episodes = self.__config['episodes']

        # Feature processor and QAgent
        self.__feature_processor = feature_processor
        self.__agent = agent

        # Number of times to repeat an action - used for frame merging
        self.__action_repeat_count = self.__config['action_repeat']

        # Create the gym enviroment
        env = gym.make(self.__config['gym_environment'])
        env.seed(0)
        self.__env = env

    def run(self):

        self.__reward = 0
        self.__reward_list = []

        for i in range(self.__episodes):
            ob = self.__env.reset()

            action = 0
            episode_reward = 0

            while True:
                # Render enviroment
                if self.__render:
                    self.__env.render()

                # Extract feature vector from observation
                ob = self.__feature_processor.extract_features(ob)
                ob = np.reshape(ob, (-1, len(ob)))

                # Let agent predict action
                # action = self.agent.predict(ob)
                action = self.__env.action_space.sample()

                # Perform action
                ob, reward, done, _ = self.__env.step(action)

                episode_reward += reward

                # Reset episode
                if done:
                    break

            self.__reward_list.append(episode_reward)

            print("Episode " + str(i) + " reward: " + str(episode_reward))

        print("Score over time: " + str(sum(self.__reward_list) / self.__episodes))

        # Clean up
        self.__env.close()

    def get_score_over_time(self):
        return sum(self.__reward_list) / self.__episodes
