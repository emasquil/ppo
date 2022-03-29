from subprocess import call
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
        key_networks: int,
        key_sampling_policy: int,
        learning_rate_params: dict,
        discount: float,
    ):
        policy_key, value_key = jax.random.split(jax.random.PRNGKey(key_networks), 2)
        self.policy_network = hk.without_apply_rng(hk.transform(policy_network))
        self.policy_params = self.policy_network.init(rng=policy_key, observations=jnp.zeros(observation_spec.shape))
        self.value_network = hk.without_apply_rng(hk.transform(value_network))
        self.value_params = self.value_network.init(rng=value_key, observations=jnp.zeros(observation_spec.shape))

        if learning_rate_params["annealing"]:
            self.learning_rate_schedule = optax.linear_schedule(
                learning_rate_params["initial_learning_rate"], 
                learning_rate_params["last_learning_rate"], 
                learning_rate_params["annealing_duration"]
            )
        else:
            self.learning_rate_schedule = learning_rate_params["initial_learning_rate"]
        self.policy_optimizer = optax.adam(self.learning_rate_schedule)
        self.policy_optimizer_state = self.policy_optimizer.init(self.policy_params)
        self.value_optimizer = optax.adam(self.learning_rate_schedule)
        self.value_optimizer_state = self.value_optimizer.init(self.value_params)

        def sampling_policy(policy_params: hk.Params, key: chex.PRNGKey, observation: np.ndarray):
            mu, sigma = self.policy_network.apply(policy_params, observation)

            action = rlax.gaussian_diagonal().sample(key, mu, sigma)
            log_prob = rlax.gaussian_diagonal().logprob(action, mu, sigma)

            return action, log_prob

        self.sampling_policy = sampling_policy
        self.sampling_keys = hk.PRNGSequence(key_sampling_policy)

        def policy(policy_params: hk.Params, observation: np.ndarray):
            return self.policy_network.apply(policy_params, observation)[0]

        self.policy = policy

        self.discount = discount

    def select_action_and_prob(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def get_learning_rate(self):
        if callable(self.learning_rate_schedule):
            return self.learning_rate_schedule(int(self.policy_optimizer_state[1].count))
        return self.learning_rate_schedule