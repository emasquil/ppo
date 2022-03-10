import numpy as np

import dm_env
from acme import Actor
import haiku as hk
import jax.numpy as jnp
from jax import tree_util
import jax
import rlax
import chex
from acme import specs
import optax
from typing import Tuple


class OnlinePPO(Actor):
    def __init__(
        self,
        observation_spec: specs.BoundedArray,
        policy_network,
        value_network,
        key: chex.PRNGKey,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
    ):
        policy_key, value_key = jax.random.split(key, 2)
        self.policy_network = hk.without_apply_rng(hk.transform(policy_network))
        self.policy_params = self.policy_network.init(
            rng=policy_key, observations=jnp.zeros(observation_spec.shape)
        )
        self.value_network = hk.without_apply_rng(hk.transform(value_network))
        self.value_params = self.policy_network.init(
            rng=value_key, observations=jnp.zeros(observation_spec.shape)
        )

        self.policy_optimizer = optax.adam(learning_rate)
        self.policy_optimizer_state = self.policy_optimizer.init(self.policy_params)
        self.value_optimizer = optax.adam(learning_rate)
        self.value_optimizer_state = self.value_optimizer.init(self.value_params)

        def greedy_policy(policy_params: hk.Params, key: chex.PRNGKey, observation: np.ndarray):
            mu, sigma = self.policy_network.apply(policy_params, observation)

            return rlax.gaussian_diagonal().sample(key, mu, sigma)

        self.greedy_policy = greedy_policy
        self.greedy_keys = hk.PRNGSequence(1)

        def policy(policy_params: hk.Params, observation: np.ndarray):
            return self.policy_network.apply(policy_params, observation)[0]

        self.policy = policy

        self.timestep = None
        self.action = None
        self.next_timestep = None
        self.discount = discount

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        # Convert to get a batch shape
        observation = tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), observation)

        action = self.greedy_policy(self.policy_params, next(self.greedy_keys), observation)

        # Convert back to single action
        action = tree_util.tree_map(lambda x: jnp.array(x).squeeze(axis=0), action)

        return action

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        self.timestep = timestep
        self.action = None
        self.next_timestep = None

    def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        assert self.timestep is None, "Please let the agent observe a first timestep."
        self.action = action

        if self.timestep.first():
            self.next_timestep = next_timestep
        else:
            self.timestep = self.next_timestep
            self.next_timestep = next_timestep

    def value_loss(
        self, value_params: hk.Params, timestep: dm_env.TimeStep, next_timestep: dm_env.TimeStep
    ) -> Tuple[chex.Array, chex.Array]:
        v_next_state = self.value_network.apply(value_params, next_timestep.observation)
        v_next_state = jax.lax.stop_gradient(v_next_state)

        v_state = self.policy_network.apply(value_params, timestep.observation)

        advantage = next_timestep.reward + self.discount * v_next_state - v_state

        return jnp.mean(jnp.square(advantage)), advantage

    def policy_loss(
        self, policy_params: hk.Params, timestep: dm_env.TimeStep, action: np.ndarray, advantage: chex.Array
    ) -> chex.Array:
        mu, sigma = self.policy_network.apply(policy_params, timestep.observation)
        action_log_probs = rlax.gaussian_diagonal().logprob(action, mu, sigma)

        return -jnp.mean(action_log_probs * advantage)

    def update(self) -> None:
        assert self.timestep is None, "Please let the agent observe a first timestep."
        assert self.action is None or self.next_timestep is None, "Please let the agent observe a timestep."

        # Update value network
        (_, advantage), value_gradients = jax.value_and_grad(self.value_loss, has_aux=True)(
            self.value_params, self.timestep, self.next_timestep
        )
        advantage = jax.lax.stop_gradient(advantage)

        value_updates, self.value_optimizer_state = self.value_optimizer.update(
            value_gradients, self.value_optimizer_state
        )
        self.value_params = optax.apply_updates(self.value_params, value_updates)

        # Update policy network
        policy_gradients = jax.grad(self.policy_loss)(self.policy_params, self.timestep, self.action, advantage)

        policy_updates, self.policy_optimizer_state = self.policy_optimizer.update(
            policy_gradients, self.policy_optimizer_state
        )
        self.policy_params = optax.apply_updates(self.policy_params, policy_updates)
