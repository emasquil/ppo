from typing import Dict

import numpy as np

import dm_env
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import clip_grads
import rlax
import optax
import haiku as hk
from acme import specs

from ppo.agents.base_agent import BaseAgent
from ppo.replay_buffers import FixedReplayBuffer


class VanillaPPO(BaseAgent):
    """Still need to code the methods:
    value_loss
    policy_loss
    update
    """

    def __init__(
        self,
        environment_spec: specs.BoundedArray,
        policy_network,
        value_network,
        key_init_networks: int,
        key_sampling_policy: int,
        key_replay_buffer: int,
        learning_rate_params: dict,
        discount: float,
        clipping_ratio_threshold: float,
        max_grad_norm: float,
    ):
        super().__init__(
            environment_spec,
            policy_network,
            value_network,
            key_init_networks,
            key_sampling_policy,
            learning_rate_params,
            discount,
        )
        self.replay_buffer = FixedReplayBuffer(key_replay_buffer)
        self.clipping_ratio_threshold = clipping_ratio_threshold
        self.max_grad_norm = max_grad_norm
        self.value_and_grad_value_loss = jax.jit(jax.value_and_grad(self.value_loss))
        self.value_and_grad_policy_loss = jax.jit(jax.value_and_grad(self.policy_loss, has_aux=True))

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        self.replay_buffer.add_first(timestep)

    def observe(self, value: float, log_probability: float, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        self.replay_buffer.add(value, log_probability, action, next_timestep)

    def add_advantages(self, advantages):
        self.replay_buffer.add_advantages(advantages)

    def value_loss(self, value_params: hk.Params, batch: Dict):
        target_value = batch["value_t"] + batch["advantage_t"]
        target_value = jax.lax.stop_gradient(target_value)
        predicted_value = self.value(value_params, batch["observation_t"])
        loss = jnp.mean(jnp.square(target_value - predicted_value))
        return loss

    def policy_loss(self, policy_params: hk.Params, batch: Dict):
        normalized_advantage_t = (batch["advantage_t"] - jnp.mean(batch["advantage_t"], axis=0)) / (
            jnp.std(batch["advantage_t"], axis=0) + 1e-5
        )
        normalized_advantage_t = jax.lax.stop_gradient(normalized_advantage_t)

        mu, sigma = self.policy(policy_params, batch["observation_t"])
        log_probability_t = rlax.gaussian_diagonal().logprob(batch["action_t"], mu, sigma)
        log_ratio = log_probability_t - jnp.squeeze(batch["log_probability_t"])
        ratio = jnp.exp(log_ratio)

        clipped_loss = jnp.minimum(
            ratio * jnp.squeeze(normalized_advantage_t),
            jax.lax.clamp(1 - self.clipping_ratio_threshold, ratio, 1 - self.clipping_ratio_threshold)
            * jnp.squeeze(normalized_advantage_t),
        )
        # Compute the KL divergence (approximation) between the old and the current policy
        kl_approximation = (ratio - 1) - log_ratio  # cf http://joschu.net/blog/kl-approx.html

        return -jnp.mean(clipped_loss), jnp.mean(kl_approximation)

    def update(self, batch: Dict):
        # Update value network

        value_loss, value_gradients = self.value_and_grad_value_loss(self.value_params, batch)
        clip_grads(value_gradients, self.max_grad_norm)
        value_updates, self.value_optimizer_state = self.value_optimizer.update(
            value_gradients, self.value_optimizer_state
        )
        self.value_params = optax.apply_updates(self.value_params, value_updates)

        # Update policy network
        (policy_loss, kl_approximation), policy_gradients = self.value_and_grad_policy_loss(
            self.policy_params,
            batch,
        )
        clip_grads(policy_gradients, self.max_grad_norm)
        policy_updates, self.policy_optimizer_state = self.policy_optimizer.update(
            policy_gradients, self.policy_optimizer_state
        )
        self.policy_params = optax.apply_updates(self.policy_params, policy_updates)

        return value_loss, policy_loss, kl_approximation

    def add_last_value(self, last_timestep):
        """Add last value for the trajectory"""
        self.replay_buffer.add_last_value(self.get_value(last_timestep.observation))

    def clear_memory(self):
        self.replay_buffer.clear_memory()

    def cast_to_numpy(self):
        self.replay_buffer.cast_to_numpy()
