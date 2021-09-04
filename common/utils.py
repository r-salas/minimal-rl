#
#
#   RL utils
#
#

import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn

from typing import Sequence, Optional, Union


def discount_rewards(rewards: Sequence, discount_rate: float = 0.99, dones: Optional[Sequence] = None,
                     next_value: Optional[float] = None):
    if dones is None:
        dones = np.zeros_like(rewards)

    if next_value is None:
        next_value = 0.0

    R = next_value
    discounted_rewards = []

    for step in reversed(range(len(rewards))):
        R = rewards[step] + discount_rate * R * (1 - dones[step])
        discounted_rewards.insert(0, R)

    return np.array(discounted_rewards)


def normalize(x, eps=np.finfo(np.float32).eps.item(), axis=None):
    return (x - np.mean(x, axis)) / (np.std(x, axis) + eps)  # eps: prevent zero std


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
    """
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def play(env: Union[str, gym.Env], predict_fn):
    if isinstance(env, str):
        env = gym.make(env)

    done = False
    obs = env.reset()

    while not done:
        env.render()
        action = predict_fn(obs)
        obs, reward, done, info = env.step(action)
