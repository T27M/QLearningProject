import gym
from gym import wrappers
import numpy as np
import time
from core.config import Config
from core.configbase import ConfigBase
from core.featureprocessor import FeatureProcessor
from qlearning.qagent import QAgent


class Environment(ConfigBase):
    def __init__(self, config: Config, feature_processor: FeatureProcessor, agent: QAgent):
        super().__init__(config)

        self.__cumulative_reward = 0
        self.__reward = 0
        self.__reward_list = []

        # Render the game enviroment
        self.__render = self._config['render']

        # Number of episodes
        self.__episodes = self._config['episodes']

        # Feature processor and QAgent
        self.__feature_processor = feature_processor
        self.__agent = agent

        self.__actions = {
            'UP': 1,
            'RIGHT': 2,
            'LEFT': 3,
            'DOWN': 4
        }

        # Number of times to repeat an action - used for frame merging
        self.__action_repeat_count = self._config['action_repeat']

        # Create the gym enviroment
        env = gym.make(self._config['gym_environment'])
        env.seed(0)
        self.__env = env

    def run(self):

        self.__reward = 0
        self.__reward_list = []

        for i in range(self.__episodes):
            print('Running Episode: ' + str(i))

            ob = self.__env.reset()

            action = 0
            episode_reward = 0
            cur_life = 3

            while True:
                # Render enviroment
                if self.__render:
                    self.__env.render()

                # Extract feature vector from observation
                state = self.__feature_processor.extract_features(ob)

                # Let agent predict action
                action = self.__agent.predict(state)

                # Perform action with action repeat
                for _ in range(self.__action_repeat_count):
                    ob, reward, done, info = self.__env.step(
                        self.__actions[action])

                    # Render enviroment
                    if self.__render:
                        self.__env.render()

                # Extract feature vector from observation
                new_state = self.__feature_processor.extract_features(ob)

                # Determine if the agent lost a life
                if(cur_life > info['ale.lives']):
                    print('Lost life')
                    cur_life = cur_life - 1
                    reward = -100

                if reward != 0:
                    # Update Q-Table
                    self.__agent.update_q_table(
                        state, new_state, action, reward)

                episode_reward += reward

                # Reset episode
                if done:
                    break

            self.__reward_list.append(episode_reward)

            episode_reward = "Episode " + \
                str(i) + " reward: " + str(episode_reward)

            with open("./data/reward.txt", "a") as myfile:
                myfile.writelines(episode_reward)

            self.__agent.save_q_table()

        print("Score over time: " + str(sum(self.__reward_list) / self.__episodes))

        # Clean up
        self.__env.close()

    def get_score_over_time(self):
        return sum(self.__reward_list) / self.__episodes
