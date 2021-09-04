#
#
#   DDPG (Deep Deterministic Policy Gradient)
#
#

import gym
import typer
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from common.logger import Logger
from common.wrappers import NormalizedEnv
from common.utils import soft_update, play, seed_everything
from common.policies import ActorContinousPolicy, CriticContinousPolicy

from itertools import chain
from collections import deque


def ddpg(env_id: str = "MountainCarContinuous-v0", num_timesteps: int = 45_000, memory_size: int = 50_000,
         batch_size: int = 128, tau: float = 1e-2, discount_rate: float = 0.99, log_frequency: int = 1_000,
         device: str = "auto", seed: int = 0):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))

    env = gym.make(env_id)
    env = NormalizedEnv(env)

    env.seed(seed)
    seed_everything(seed)

    actor_policy = ActorContinousPolicy(env.observation_space, env.action_space).to(device)
    critic_policy = CriticContinousPolicy(env.observation_space, env.action_space).to(device)
    target_actor_policy = ActorContinousPolicy(env.observation_space, env.action_space).to(device)
    target_critic_policy = CriticContinousPolicy(env.observation_space, env.action_space).to(device)

    for param in chain(target_actor_policy.parameters(), target_critic_policy.parameters()):
        param.requires_grad = False

    target_actor_policy.load_state_dict(actor_policy.state_dict())
    target_critic_policy.load_state_dict(critic_policy.state_dict())

    actor_optimizer = optim.Adam(actor_policy.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic_policy.parameters(), lr=1e-3)

    noise_gen = OUActionNoise(0.0 * np.ones(env.action_space.shape), 0.2 * np.ones(env.action_space.shape))

    memory = deque(maxlen=memory_size)

    logger = Logger(log_frequency=log_frequency)

    timestep = 0
    obs = env.reset()

    while timestep < num_timesteps:
        with torch.no_grad():
            action = actor_policy(t(obs).float())

        action = action.cpu().numpy() + noise_gen()

        next_obs, reward, done, info = env.step(action)

        memory.append((obs, action, reward, done, next_obs))

        logger.log_step(reward, done)

        timestep += 1

        obs = next_obs

        if done:
            obs = env.reset()

        if len(memory) >= batch_size:
            batch_obs, batch_action, batch_reward, batch_done, batch_next_obs = zip(*random.sample(memory, batch_size))

            current_q_vals = critic_policy(t(batch_obs), t(batch_action)).squeeze()
            next_actions = target_actor_policy(t(batch_next_obs))
            next_q_vals = target_critic_policy(t(batch_next_obs), t(next_actions)).squeeze()
            target_q_vals = t(batch_reward) + (1 - t(batch_done).int()) * discount_rate * next_q_vals

            critic_loss = F.mse_loss(current_q_vals, target_q_vals)

            critic_optimizer.zero_grad()

            critic_loss.backward()

            critic_optimizer.step()

            actor_loss = -critic_policy(t(batch_obs), actor_policy(t(batch_obs))).mean()

            actor_optimizer.zero_grad()

            actor_loss.backward()

            actor_optimizer.step()

            logger.log_metric("loss/actor", actor_loss.item())
            logger.log_metric("loss/critic", critic_loss.item())

            soft_update(actor_policy, target_actor_policy, tau)
            soft_update(critic_policy, target_critic_policy, tau)

    def predict(obs):
        return target_actor_policy(t(obs)).cpu().numpy()

    play(env_id, predict)


# Ornstein-Ulhenbeck Process
class OUActionNoise:
    """Adapted from https://keras.io/examples/rl/ddpg_pendulum/"""

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        if x_initial is None:
            x_initial = np.zeros_like(mean)

        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = x_initial

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial


if __name__ == "__main__":
    typer.run(ddpg)
