#
#
#   A2C (Advantage actor critic)
#
#

import gym
import typer
import torch
import torch.optim as optim

from common.logger import Logger
from common.wrappers import DiscretePendulum
from common.policies import ActorCriticDiscretePolicy
from common.utils import discount_rewards, seed_everything, play

from torch.distributions.categorical import Categorical


def a2c(env_id="Pendulum-v0", max_timesteps: int = 500_000, num_envs: int = 8, asynchronous: bool = True,
        discount_rate: float = 0.99, n_steps: int = 5, value_coeff: float = 0.5, entropy_coef: float = 0.0,
        learning_rate: float = 1e-4, log_frequency: int = 1_000, device: str = "auto", seed: int = 0,
        test: bool = True):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))

    wrappers = None

    if env_id == "Pendulum-v0":
        wrappers = DiscretePendulum

    env = gym.vector.make(env_id, num_envs=num_envs, asynchronous=asynchronous, wrappers=wrappers)

    # Configure seed
    env.seed(seed)
    seed_everything(seed)

    policy = ActorCriticDiscretePolicy(env.single_observation_space, env.single_action_space).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    logger = Logger(log_frequency, num_envs=num_envs)

    timestep = 0
    obs = env.reset()

    n_steps_dones = []
    n_steps_rewards = []
    n_steps_entropy = 0.0
    n_steps_state_values = []
    n_steps_log_prob_actions = []

    while timestep < max_timesteps:
        state_value, action_logits = policy(t(obs))

        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()

        next_obs, reward, done, info = env.step(action.cpu().numpy())

        n_steps_dones.append(done)
        n_steps_rewards.append(reward)
        n_steps_state_values.append(state_value)
        n_steps_entropy += action_dist.entropy().mean()
        n_steps_log_prob_actions.append(action_dist.log_prob(action))

        timestep += 1

        logger.log_step(reward, done)

        if (timestep % n_steps) == 0:
            next_state_value, _ = policy(t(next_obs))

            with torch.no_grad():
                discounted_rewards = discount_rewards(n_steps_rewards, discount_rate, n_steps_dones,
                                                      next_state_value.squeeze().cpu().numpy())

            advantage = t(discounted_rewards) - torch.stack(n_steps_state_values).view(-1, num_envs)

            critic_loss = advantage.pow(2).mean()

            actor_loss = -torch.sum(torch.stack(n_steps_log_prob_actions) * advantage.detach())

            loss = actor_loss + critic_loss * value_coeff + n_steps_entropy * entropy_coef

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            n_steps_entropy = 0.0
            n_steps_dones.clear()
            n_steps_rewards.clear()
            n_steps_state_values.clear()
            n_steps_log_prob_actions.clear()

            logger.log_metric("critic_loss", critic_loss.item())
            logger.log_metric("actor_loss", actor_loss.item())
            logger.log_metric("entropy", n_steps_entropy)
            logger.log_metric("loss", loss.item())

        obs = next_obs

    if test:
        policy.eval()

        def predict(obs):
            return torch.argmax(policy(t(obs))[1]).cpu().numpy()

        env = gym.make(env_id)

        if env_id == "Pendulum-v0":
            env = DiscretePendulum(env)

        play(env, predict)


if __name__ == "__main__":
    typer.run(a2c)
