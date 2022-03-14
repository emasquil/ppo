import numpy as np

import dm_env
import haiku as hk
import jax.numpy as jnp
from jax import tree_util
import jax
import rlax
import chex
from acme import specs
import optax
from typing import Tuple

from ppo.agents.base_agent import BaseAgent


class OnlinePPO(BaseAgent):
    def __init__(
        self,
        observation_spec: specs.BoundedArray,
        policy_network,
        value_network,
        key: chex.PRNGKey,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
    ):
        super(OnlinePPO, self).__init__(observation_spec, policy_network, value_network, key, learning_rate, discount)

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        # Convert to get a batch shape
        observation = tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), timestep.observation
        )
        timestep = dm_env.TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            observation=observation,
            discount=timestep.discount,
        )

        self.timestep = timestep
        self.action = None
        self.next_timestep = None

    def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        assert (
            self.timestep is not None
        ), "Please let the agent observe a first timestep."
        # Convert to get a batch shape
        observation = tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), next_timestep.observation
        )
        next_timestep = dm_env.TimeStep(
            step_type=next_timestep.step_type,
            reward=next_timestep.reward,
            observation=observation,
            discount=next_timestep.discount,
        )

        self.action = action

        if self.timestep.first():
            self.next_timestep = next_timestep
        else:
            self.timestep = self.next_timestep
            self.next_timestep = next_timestep

    def value_loss(
        self,
        value_params: hk.Params,
        timestep: dm_env.TimeStep,
        next_timestep: dm_env.TimeStep,
    ) -> Tuple[chex.Array, chex.Array]:
        v_next_state = self.value_network.apply(value_params, next_timestep.observation)
        v_next_state = jax.lax.stop_gradient(v_next_state)

        v_state = self.value_network.apply(value_params, timestep.observation)

        advantage = next_timestep.reward + self.discount * v_next_state - v_state

        return jnp.mean(jnp.square(advantage)), advantage

    def policy_loss(
        self,
        policy_params: hk.Params,
        timestep: dm_env.TimeStep,
        action: np.ndarray,
        advantage: chex.Array,
    ) -> chex.Array:
        mu, sigma = self.policy_network.apply(policy_params, timestep.observation)
        action_log_probs = rlax.gaussian_diagonal().logprob(action, mu, sigma)

        return -jnp.mean(action_log_probs * advantage)

    def update(self) -> None:
        assert (
            self.timestep is not None
        ), "Please let the agent observe a first timestep."
        assert (
            self.action is not None and self.next_timestep is not None
        ), "Please let the agent observe a timestep."

        # Update value network
        (_, advantage), value_gradients = jax.value_and_grad(
            self.value_loss, has_aux=True
        )(self.value_params, self.timestep, self.next_timestep)
        advantage = jax.lax.stop_gradient(advantage)

        value_updates, self.value_optimizer_state = self.value_optimizer.update(
            value_gradients, self.value_optimizer_state
        )
        self.value_params = optax.apply_updates(self.value_params, value_updates)

        # Update policy network
        policy_gradients = jax.grad(self.policy_loss)(
            self.policy_params, self.timestep, self.action, advantage
        )

        policy_updates, self.policy_optimizer_state = self.policy_optimizer.update(
            policy_gradients, self.policy_optimizer_state
        )
        self.policy_params = optax.apply_updates(self.policy_params, policy_updates)
