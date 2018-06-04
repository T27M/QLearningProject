import gym
from gym import wrappers
import numpy as np
import time
import json
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

        self.__evaluation = self._config['eval']

        # Render the game enviroment
        self.__render = self._config['render']

        # Number of episodes
        self.__episodes = self._config['episodes']

        # Feature processor and QAgent
        self.__feature_processor = feature_processor
        self.__agent = agent

        self.__actions = self._config['actions']

        # Number of times to repeat an action - used for frame merging
        self.__action_repeat_count = self._config['action_repeat']

        self.__env_is_pacman = self._config['gym_environment'] == 'MsPacman-v0'

        # Create the gym enviroment
        env = gym.make(self._config['gym_environment'])
        env.seed(0)
        self.__env = env

    def run(self):

        self.__reward = 0
        self.__reward_list = []

        for i in range(self.__episodes):
            print('Running Episode: ' + str(i))

            s = self.__env.reset()

            if self.__env_is_pacman:
                # Extract feature vector from observation
                s = self.__feature_processor.extract_features(s)
            else:
                s = np.around(s, decimals=3)

            action = 0
            episode_reward = 0
            cur_life = 3

            while True:
                # Render enviroment
                if self.__render:
                    self.__env.render()

                # Let agent predict action
                action = self.__agent.predict(s)

                # # Perform action with action repeat
                # for _ in range(self.__action_repeat_count):
                s1, reward, done, info = self.__env.step(action)

                if self.__env_is_pacman:
                    # Extract feature vector from observation
                    s1 = self.__feature_processor.extract_features(s1)
                else:
                    s1 = np.around(s1, decimals=3)

                if self.__env_is_pacman:
                    # Determine if the agent lost a life
                    if(cur_life > info['ale.lives']):
                        print('Lost life')
                        cur_life = cur_life - 1
                        reward = -100

                if not self.__evaluation:
                    # Update Q-Table
                    self.__agent.update_q_table(s, s1, action, reward)

                # Pass along state
                s = s1

                episode_reward += reward

                # Reset episode
                if done:
                    break

            er_dict = {
                'episode': i,
                'episode_reward': episode_reward
            }

            self.__reward_list.append(er_dict)

            print("\t Episode reward: " + str(episode_reward))

            if not self.__evaluation:
                self.__agent.save_q_table()

        with open(self._data_dir + 'qt.reward.json', "w") as file:
            json.dump(self.__reward_list, file)

        x = np.asarray([d['episode_reward'] for d in self.__reward_list])

        print("Best reward: " + str(np.max(x)))

        # Clean up
        self.__env.close()

    def get_score_over_time(self):
        return sum(self.__reward_list) / self.__episodes
