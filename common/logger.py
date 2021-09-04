#
#
#   RL Stats
#
#

import time
import numpy as np

from typing import Dict
from collections import defaultdict


class Logger:

    def __init__(self, log_frequency: int = 100, num_envs: int = 1):
        self.num_envs = num_envs
        self.log_frequency = log_frequency

        self._episode = 0
        self._timestep = 0
        self._ep_steps = np.zeros(num_envs, dtype=int)
        self._ep_rewards = np.zeros(num_envs, dtype=float)
        self._start_time = time.time()
        self._metrics = defaultdict(list)

    def log_step(self, reward, done):
        self._timestep += 1

        self._ep_steps += 1
        self._ep_rewards += reward

        done_envs = np.flatnonzero(done)

        for env_idx in done_envs:
            self._metrics["episode/length"].append(self._ep_steps[env_idx])
            self._metrics["episode/reward"].append(self._ep_rewards[env_idx])

            self._episode += 1

        self._ep_rewards[done_envs] = 0
        self._ep_steps[done_envs] = 0

        if (self._timestep % self.log_frequency) == 0:
            print(f"time/timestep: {self._timestep}")
            print(f"time/episode: {self._episode}")
            print(f"time/elapsed: {time.time() - self._start_time:.2f}s")

            for metric_name, metric_values in self._metrics.items():
                print(f"{metric_name}: {np.mean(metric_values):.2f}")

            self._metrics.clear()

            print("*" * 20)

    def log_metric(self, name: str, value: float):
        self._metrics[name].append(value)
