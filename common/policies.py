#
#
#
#   Policies
#
#

import torch
import gym.spaces
import torch.nn as nn


class QNetworkDiscretePolicy(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

    def forward(self, obs):
        return self.fc(obs)


class ActorCriticDiscretePolicy(nn.Module):

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

    def forward(self, obs):
        x = self.common(obs)
        return self.critic(x), self.actor(x)


class ActorDiscretePolicy(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)  # logits
        )

    def forward(self, obs):
        x = self.fc(obs)
        return x


class ActorContinousPolicy(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.shape[0]),
        )

    def forward(self, obs):
        x = self.fc(obs)
        x = torch.tanh(x)
        return x


class CriticDiscretePolicy(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0] + action_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)  # logits
        )

    def forward(self, obs, action):
        if obs.ndim < 2:
            obs = obs.unsqueeze(-1)
        if action.ndim < 2:
            action = action.unsqueeze(-1)

        x = torch.cat([obs, action], 1)
        x = self.fc(x)
        return x


class CriticContinousPolicy(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0] + action_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        if obs.ndim < 2:
            obs = obs.unsqueeze(-1)
        if action.ndim < 2:
            action = action.unsqueeze(-1)

        x = torch.cat([obs, action], 1)
        x = self.fc(x)
        return x
