from linearfa.lfaqagent import LfaQAgent
from linearfa.env import Env
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--episodes',
                    help='# Number of episodes',
                    type=int)

parser.add_argument('-t', '--train',
                    help='train the agent',
                    action='store_true')

parser.add_argument('-lr', '--learning_rate',
                    help='learning rate of the agent e.g. 0.0001',
                    type=float)

parser.add_argument('-d', '--decay',
                    help='decay the learning rate',
                    action='store_true')

parser.add_argument('-lw', '--load-weights',
                    help='load saved weights for the agent',
                    type=str)

parser.add_argument('-r', '--render',
                    help='render the environment',
                    action='store_true')

args = parser.parse_args()

episodes = args.episodes
train = args.train
learning_rate = args.learning_rate

decay = args.decay
load_weights = args.load_weights
render = args.render

print("Episodes: " + str(episodes))
print("Train: " + str(train))
print("Learning Rate: " + str(learning_rate))

print("Learning Rate Decay: " + str(decay))

path = './data/lfa/' + load_weights + '/'
print("Loading weights: " + path)

print("Render Environment: " + str(render))

input('Press any key to continue...')

q = LfaQAgent(learning_rate)

if load_weights is not None:

    q.load_weights(path)

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
