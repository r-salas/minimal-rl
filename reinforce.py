#
#
#   REINFORCE a.k.a Vanilla Policy Gradient
#
#

import gym
import typer
import torch
import numpy as np
import torch.optim as optim

from common.stats import Stats
from common.policies import ActorPolicy
from common.utils import discount_rewards, normalize

from torch.distributions.categorical import Categorical


def reinforce(env_id="CartPole-v1", max_timesteps: int = 100_000, discount_rate: float = 0.9,
              log_frequency: int = 1_000, device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(x):
        return torch.as_tensor(x, device=torch.device(device))

    env = gym.make(env_id)

    policy = ActorPolicy(env.observation_space, env.action_space)

    optimizer = optim.Adam(policy.parameters())

    stats = Stats(log_frequency)

    timestep = 0
    obs = env.reset()

    ep_rewards = []
    ep_log_prob_actions = []

    while timestep < max_timesteps:
        action_logits = policy(t(obs).float())
        action_dist = Categorical(logits=action_logits)

        action = action_dist.sample()

        obs, reward, done, info = env.step(action.item())

        ep_rewards.append(reward)
        ep_log_prob_actions.append(action_dist.log_prob(action))

        stats.step(reward, done)

        timestep += 1

        if done:
            discounted_rewards = normalize(discount_rewards(ep_rewards, discount_rate))

            loss = -torch.sum(t(discounted_rewards) * torch.stack(ep_log_prob_actions))  # negative: gradient ascent

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            obs = env.reset()

            ep_rewards.clear()
            ep_log_prob_actions.clear()

            stats.log("loss", loss.item())


if __name__ == "__main__":
    typer.run(reinforce)
