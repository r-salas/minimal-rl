#
#
#   DQN
#
#

import gym
import typer
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from common.stats import Stats
from common.policies import QNetworkPolicy


def dqn(env_id="CartPole-v1", max_timesteps: int = 200_000, discount_rate: float = 0.95, batch_size: int = 32,
        train_frequency: int = 8, replay_buffer_size: int = 1_000, exploration_fraction: float = 0.2,
        target_update_frequency: int = 1_000, log_frequency: int = 1_000, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(x):
        return torch.as_tensor(x, device=torch.device(device))

    env = gym.make(env_id)

    online_policy = QNetworkPolicy(env.observation_space, env.action_space).to(device)
    target_policy = QNetworkPolicy(env.observation_space, env.action_space).to(device)

    for param in target_policy.parameters():
        param.requires_grad = False

    online_policy.train()
    target_policy.eval()

    optimizer = optim.RMSprop(online_policy.parameters())

    replay_buffer = deque(maxlen=replay_buffer_size)

    stats = Stats(log_frequency=log_frequency)

    timestep = 0
    obs = env.reset()

    while True:
        if np.random.rand() < exploration_fraction:
            action = env.action_space.sample()
        else:
            action_logits = target_policy(t(obs).float())
            action = action_logits.argmax().item()

        next_obs, reward, done, info = env.step(action)

        replay_buffer.append((obs, action, next_obs, reward, done))

        timestep += 1

        stats.step(reward, done)

        if done:
            obs = env.reset()
        else:
            obs = next_obs

        if (timestep % train_frequency) == 0 and len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_dones = zip(*batch)

            q_values = online_policy(t(batch_obs).float())

            q_values_next_obs = target_policy(t(batch_next_obs).float())

            next_q_values = q_values_next_obs.max(1).values

            current_q_values = q_values.gather(1, t(batch_actions).view(-1, 1)).squeeze()

            expected_q_values = t(batch_rewards) + (1 - t(batch_dones).int()) * discount_rate * next_q_values

            loss = F.mse_loss(current_q_values, expected_q_values)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            stats.log("train/loss", loss.item())

        if (timestep % target_update_frequency) == 0:
            target_policy.load_state_dict(online_policy.state_dict())

        if timestep >= max_timesteps:
            break


if __name__ == "__main__":
    typer.run(dqn)
