import gym
import numpy as np
from qlearning.qagent import QAgent
from core.config import Config

config = Config('example-config.json')

q_agent = QAgent(config)

env = gym.make('CartPole-v0')
env.reset()

rewards = []

q_agent.load_q_table()

for i in range(5000):

    print('Episode: ' + str(i))

    episode_reward = 0
    action = 0
    s = np.around(env.reset(), decimals=2)

    while(True):

        env.render()

        # Predict action
        action = q_agent.predict(s)

        # Take action
        s1, reward, done, info = env.step(action)
        s1 = np.around(s1, decimals=1)

        # Update QTable
        q_agent.update_q_table(s, s1, action, reward)

        # Pass state along
        s = s1

        # Total episode reward
        episode_reward += reward

        if done:
            q_agent.save_q_table()

            print("Episode reward: " + str(episode_reward))
            rewards.append(episode_reward)

            with open("./data/reward.txt", "a") as myfile:
                myfile.writelines("Episode: " + str(i) +
                                  " | Reward: " + str(episode_reward) + "\n")

            break
