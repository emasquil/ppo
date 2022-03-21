from typing import Tuple
import numpy as np

from acme import Actor
import haiku as hk
import jax.numpy as jnp
from jax import tree_util
import jax
import rlax
import chex
from acme import specs
import optax


class BaseAgent(Actor):
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
        self.policy_params = self.policy_network.init(rng=policy_key, observations=jnp.zeros(observation_spec.shape))
        self.value_network = hk.without_apply_rng(hk.transform(value_network))
        self.value_params = self.value_network.init(rng=value_key, observations=jnp.zeros(observation_spec.shape))

        self.policy_optimizer = optax.adam(learning_rate)
        self.policy_optimizer_state = self.policy_optimizer.init(self.policy_params)
        self.value_optimizer = optax.adam(learning_rate)
        self.value_optimizer_state = self.value_optimizer.init(self.value_params)

        def sampling_policy(policy_params: hk.Params, key: chex.PRNGKey, observation: np.ndarray):
            mu, sigma = self.policy_network.apply(policy_params, observation)

            action = rlax.gaussian_diagonal().sample(key, mu, sigma)
            log_prob = rlax.gaussian_diagonal().logprob(action, mu, sigma)

            return action, log_prob

        self.sampling_policy = sampling_policy
        self.sampling_keys = hk.PRNGSequence(1)

        def policy(policy_params: hk.Params, observation: np.ndarray):
            return self.policy_network.apply(policy_params, observation)[0]

        self.policy = policy

        self.discount = discount

    def select_action_and_prob(self, observation: np.ndarray) -> Tuple[np.ndarray]:
        # Convert to get a batch shape
        observation = tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), observation)

        action, log_prob = self.sampling_policy(self.policy_params, next(self.sampling_keys), observation)

        # Convert back to single action
        action = tree_util.tree_map(lambda x: jnp.array(x).squeeze(axis=0), action)

        return action, log_prob

    def get_value(self, observation: np.ndarray) -> np.ndarray:
        observation = tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), observation)
        return self.value_network.apply(self.value_params, observation)

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        # Convert to get a batch shape
        observation = tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), observation)

        action = self.policy(self.policy_params, observation)
        # Convert back to single action
        action = tree_util.tree_map(lambda x: jnp.array(x).squeeze(axis=0), action)
        return action
