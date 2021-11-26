#
#
#   REINFORCE a.k.a Vanilla Policy Gradient
#
#

import gym
import typer
import torch
import torch.optim as optim

from common.logger import Logger
from common.policies import ActorDiscretePolicy
from common.utils import discount_rewards, normalize, seed_everything, play

from torch.distributions.categorical import Categorical


def reinforce(env_id="CartPole-v1", max_timesteps: int = 150_000, discount_rate: float = 0.99,
              learning_rate: float = 1e-3, log_frequency: int = 1_000, device: str = "auto", seed: int = 0,
              test: bool = True):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))

    env = gym.make(env_id)

    env.seed(seed)
    seed_everything(seed)

    policy = ActorDiscretePolicy(env.observation_space, env.action_space)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    logger = Logger(log_frequency)

    timestep = 0
    obs = env.reset()

    ep_rewards = []
    ep_log_prob_actions = []

    while timestep < max_timesteps:
        action_logits = policy(t(obs))
        action_dist = Categorical(logits=action_logits)

        action = action_dist.sample()

        obs, reward, done, info = env.step(action.item())

        ep_rewards.append(reward)
        ep_log_prob_actions.append(action_dist.log_prob(action))

        logger.log_step(reward, done)

        timestep += 1

        if done:
            discounted_rewards = normalize(discount_rewards(ep_rewards, discount_rate=discount_rate))

            loss = -torch.sum(t(discounted_rewards) * torch.stack(ep_log_prob_actions))  # negative: gradient ascent

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            obs = env.reset()

            ep_rewards.clear()
            ep_log_prob_actions.clear()

            logger.log_metric("loss", loss.item())

    if test:
        policy.eval()

        def predict(obs):
            return torch.argmax(policy(t(obs))).cpu().numpy()

        play(env, predict)


if __name__ == "__main__":
    typer.run(reinforce)
