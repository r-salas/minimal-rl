#
#
#
#   Policies
#
#

import gym.spaces
import torch.nn as nn


class QNetworkPolicy(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

    def forward(self, x):
        return self.fc(x)


class ActorCriticPolicy(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete):
        super().__init__()

        self.common = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.actor = nn.Linear(64, action_space.n)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.common(x)
        return self.critic(x), self.actor(x)


class ActorPolicy(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)  # logits
        )

    def forward(self, x):
        x = self.fc(x)
        return x
