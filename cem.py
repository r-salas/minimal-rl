#
#
#   Cross entropy method (CEM)
#
#

import gym
import math
import torch
import typer
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

from common.logger import Logger
from common.utils import seed_everything, play
from common.policies import ActorDiscretePolicy


def cem(env_id: str = "Acrobot-v1", max_timesteps: int = 2_000_000, batch_size: int = 8,
        elite_fraction: float = 0.05, log_frequency: int = 10_000, discount_rate: float = 0.99,
        seed: int = 0, device: str = "auto", test: bool = True):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))

    elite_episodes = math.floor(batch_size * elite_fraction)

    env = gym.make(env_id)

    env.seed(seed)
    seed_everything(seed)

    policy = ActorDiscretePolicy(env.observation_space, env.action_space)

    optimizer = optim.Adam(policy.parameters())

    logger = Logger(log_frequency)

    timestep = 0
    obs = env.reset()

    batch = []

    episode_step = 0
    episode_states = []
    episode_actions = []
    episode_discounted_reward = 0.0

    while timestep < max_timesteps:
        action_logits = policy(t(obs))
        action_dist = Categorical(logits=action_logits)

        action = action_dist.sample()

        next_obs, reward, done, info = env.step(action.item())

        logger.log_step(reward, done)

        timestep += 1

        episode_step += 1
        episode_states.append(obs)
        episode_actions.append(action.item())
        episode_discounted_reward += reward * (discount_rate ** episode_step)

        obs = next_obs

        if done:
            batch.append((episode_states, episode_actions, episode_discounted_reward))

            episode_step = 0
            episode_states = []
            episode_actions = []
            episode_discounted_reward = 0.0

            obs = env.reset()

            if len(batch) >= batch_size:
                batch_states, batch_actions, batch_rewards = zip(*batch)

                elite_episode_indices = np.argsort(batch_rewards)[-elite_episodes:]
                elite_batch_states = [batch_states[i] for i in elite_episode_indices]
                elite_batch_actions = [batch_actions[i] for i in elite_episode_indices]

                pred_action_logits = policy(t(np.vstack(elite_batch_states)))
                target_actions = t(np.concatenate(elite_batch_actions))
                loss = F.cross_entropy(pred_action_logits, target_actions.long())

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                batch.clear()

    if test:
        policy.eval()

        def predict(obs):
            return torch.argmax(policy(t(obs))).cpu().numpy()

        play(env_id, predict)


if __name__ == "__main__":
    typer.run(cem)
