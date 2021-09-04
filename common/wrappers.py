#
#
#   Wrappers
#
#
import math

import gym


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b


class DiscretePendulum(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(n=9)

    def action(self, action):
        return [-2 + action * 0.5]
