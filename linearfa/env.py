import sys
import gym
from core.config import Config
from core.featureprocessor import FeatureProcessor


class Env(object):
    def __init__(self, agent, render, train, decay):
        self.__render = render
        self.__train = train
        self.__decay = decay
        self.__agent = agent
        self.__player_control = False

        self.__starting_lives = 3
        self.__current_lives = 3

        if train:
            print("\tTraining agent...")
        else:
            print("\tUsing policy...")

        print('\tUsing weights..\n')
        print(self.__agent.weights())

    def run(self, episodes):

        env = gym.make('CartPole-v0')
        # env = gym.make('MsPacman-v0')
        env.reset()

        config = Config()
        fp = FeatureProcessor(config)

        for episode in range(episodes):
            print('\n')
            print('Episode: ' + str(episode))
            print("\tLearning Rate:" + str(self.__agent.alpha))

            episode_reward = 0
            action = -1

            s = env.reset()

            # Agent has control only when play is valid i.e it can move
            # print('Awaiting control...')
            # for _ in range(80):
            #     if self.__render:
            #         env.render()
            #     s, reward, done, _ = env.step(0)
            # print('Agent has control!')

            # s = fp.extract_features(s)

            while(True):
                if self.__render:
                    env.render()

                if self.__train:
                    # Predict action
                    action = self.__agent.predict(s)
                    # print('Training action: ' + str(action))
                else:
                    # Use policy
                    action = self.__agent.act(s)
                    # print('Policy action: ' + str(action))

                # Take action
                s1, reward, done, info = env.step(action)
                # s1 = fp.extract_features(s1)

                # Pass state along
                s = s1

                # if info['ale.lives'] < self.__current_lives:
                #     self.__current_lives = self.__current_lives - 1
                #     print('Lost life')
                #     reward = -100

                if self.__train and not self.__player_control:
                    # Update QTable
                    self.__agent.update_fa(s, action, s1, reward)

                if not self.__player_control:
                    # Total episode reward
                    episode_reward += reward

                if done:
                    self.__current_lives = self.__starting_lives

                    if not self.__player_control:
                        print("\tEpisode reward: " + str(episode_reward))
                        self.__agent.add_reward(episode, episode_reward)

                        print(self.__agent.weights())

                    if self.__train and not self.__player_control:
                        print('\tSaving weights...')
                        self.__agent.save_weights()

                        if self.__decay:
                            if self.__agent.decay_learning_rate(episode):
                                print('\tDecay learning rate..')

                    print('\n ---')
                    break
