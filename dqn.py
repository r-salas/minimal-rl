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

from common.logger import Logger
from common.utils import seed_everything, play
from common.policies import QNetworkDiscretePolicy


def dqn(env_id="LunarLander-v2", max_timesteps: int = 250_000, discount_rate: float = 0.99, batch_size: int = 64,
        train_frequency: int = 16, replay_buffer_size: int = 10_000, exploration_fraction: float = 0.2,
        exploration_initial_eps: float = 1.0, exploration_final_eps: float = 0.1, target_update_frequency: int = 600,
        learning_rate: float = 4e-3, log_frequency: int = 1_000, device="auto", seed: int = 0, test: bool = True):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))

    env = gym.make(env_id)

    env.seed(seed)
    seed_everything(seed)
    env.action_space.np_random.seed(seed)

    online_policy = QNetworkDiscretePolicy(env.observation_space, env.action_space).to(device)
    target_policy = QNetworkDiscretePolicy(env.observation_space, env.action_space).to(device)

    for param in target_policy.parameters():
        param.requires_grad = False

    target_policy.load_state_dict(online_policy.state_dict())

    online_policy.train()
    target_policy.eval()

    optimizer = optim.Adam(online_policy.parameters(), lr=learning_rate)

    replay_buffer = deque(maxlen=replay_buffer_size)

    exploration = EpsilonExploration(int(max_timesteps * exploration_fraction), exploration_initial_eps,
                                     exploration_final_eps)

    logger = Logger(log_frequency=log_frequency)

    timestep = 0
    obs = env.reset()

    while timestep < max_timesteps:
        if exploration():
            action = env.action_space.sample()
        else:
            action_logits = target_policy(t(obs))
            action = action_logits.argmax().item()

        next_obs, reward, done, info = env.step(action)

        replay_buffer.append((obs, action, next_obs, reward, done))

        timestep += 1

        logger.log_step(reward, done)

        if done:
            obs = env.reset()
        else:
            obs = next_obs

        if (timestep % train_frequency) == 0 and len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_dones = zip(*batch)

            q_values = online_policy(t(batch_obs))

            q_values_next_obs = target_policy(t(batch_next_obs))

            next_q_values = q_values_next_obs.max(1).values

            current_q_values = q_values.gather(1, t(batch_actions).long().view(-1, 1)).squeeze()

            expected_q_values = t(batch_rewards) + (1 - t(batch_dones).int()) * discount_rate * next_q_values

            loss = F.mse_loss(current_q_values, expected_q_values)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            logger.log_metric("train/loss", loss.item())
            logger.log_metric("train/epsilon", exploration.epsilon)

        if (timestep % target_update_frequency) == 0:
            target_policy.load_state_dict(online_policy.state_dict())

    if test:
        target_policy.eval()

        def predict(obs):
            return torch.argmax(target_policy(t(obs))).cpu().numpy()

        play(env, predict)


class EpsilonExploration:

    def __init__(self, exploration_timesteps, exploration_initial_eps, exploration_final_eps):
        self.exploration_timesteps = exploration_timesteps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps

        self.epsilon = exploration_initial_eps
        self._reduce_rate = (exploration_initial_eps - exploration_final_eps) / exploration_timesteps

    def __call__(self):
        should_explore = np.random.rand() < self.epsilon
        self.epsilon = max(self.exploration_final_eps, self.epsilon - self._reduce_rate)
        return should_explore


if __name__ == "__main__":
    typer.run(dqn)
