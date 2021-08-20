#
#
#   RL utils
#
#

import numpy as np

from typing import Sequence


def discount_rewards(rewards: Sequence, discount_rate: float = 0.99):
    r = np.array([discount_rate**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def normalize(x, eps=np.finfo(np.float32).eps.item()):
    return (x - np.mean(x)) / (np.std(x) + eps)  # eps: prevent zero std
