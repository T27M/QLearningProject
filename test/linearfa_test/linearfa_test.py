import unittest
import numpy as np
from core.config import Config
from linearfa.lfaqagent import LfaQAgent


class LinearFaTest(unittest.TestCase):
    def test_given_state_and_weights_correctly_update_weights(self):

        learning_rate = 0.1
        discount_factor = 0.9

        action = 1

        s = [0.012, -0.017, 0, -0.009]
        s1 = [-0.041 - 0.047 - 0.036 - 0.034]

        reward = 10

        lfa = LfaQAgent(learning_rate=learning_rate,
                        discount_factor=discount_factor, environment=None)

        # Calculate update
        # ∆wi = α(r + γ max a'Q̃(s', a', wi) − Q̃(s, a, wi)) δQ̃(s, a, w)/ δwi

        weight_change = learning_rate * ()

        print(np.sum(q_mult, dtype=np.float64))
        print(q_dot)


if __name__ == '__main__':
    unittest.main()
