from linearfa.lfaqagent import LfaQAgent
from linearfa.env import Env

render = False
train = True
decay = False
episodes = 20

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
