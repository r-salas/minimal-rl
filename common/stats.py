#
#
#   RL Stats
#
#

import time
import numpy as np

from collections import defaultdict


class Stats:

    def __init__(self, log_frequency: int = 100):
        self.log_frequency = log_frequency

        self._episode = 0
        self._timestep = 0
        self._ep_rewards = []
        self._start_time = time.time()
        self._metrics = defaultdict(list)

    def step(self, reward, done):
        self._timestep += 1

        self._ep_rewards.append(reward)

        if done:
            self._metrics["episode/length"].append(len(self._ep_rewards))
            self._metrics["episode/reward"].append(np.sum(self._ep_rewards))

            self._episode += 1
            self._ep_rewards.clear()

        if (self._timestep % self.log_frequency) == 0:
            print(f"time/timestep: {self._timestep}")
            print(f"time/episode: {self._episode}")
            print(f"time/elapsed: {time.time() - self._start_time:.2f}s")

            for metric_name, metric_values in self._metrics.items():
                print(f"{metric_name}: {np.mean(metric_values):.2f}")

            self._metrics.clear()

            print("*" * 20)

    def log(self, name, value):
        self._metrics[name].append(value)
