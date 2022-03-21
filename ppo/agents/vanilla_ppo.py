import numpy as np

import dm_env
import chex
import haiku as hk 
import jax.numpy as jnp
import rlax


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
        observation_spec: specs.BoundedArray,
        policy_network,
        value_network,
        key: chex.PRNGKey,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
    ):
        super(VanillaPPO, self).__init__(observation_spec, policy_network, value_network, key, learning_rate, discount)
        self.replay_buffer = FixedReplayBuffer(buffer_capacity=50)

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        self.replay_buffer.add_first(timestep)

    def observe(self, value: float, log_probability: float, action: np.ndarray, next_timestep: dm_env.TimeStep, advantage: float) -> None:
        self.replay_buffer.add(value, log_probability, action, next_timestep, advantage)

    def update(self):
        pass

    def prob_ratio(
        self,
        policy_params: hk.Params,
        observation: chex.Array, 
        action: chex.Array, 
        action_log_probs_old: chex.Array    
    ) -> chex.Array:
        mu, sigma = self.policy_network.apply(policy_params, observation)
        action_log_probs = rlax.gaussian_diagonal().logprob(action, mu, sigma)
        ratio = jnp.exp( (action_log_probs) - (action_log_probs_old))
        return ratio

    def normalize_advantage(self, advantage: chex.Array) -> chex.Array:
        mean = jnp.mean(advantage)
        std = jnp.std(advantage)
        normalized_advantage = (advantage - mean)/(std + 1e-8) # add a little jitter to avoid division by 0
        return normalized_advantage
    
    def policy_loss(
        self,
        policy_params: hk.Params,
        observation,
        action: np.ndarray,
        advantage: chex.Array,
        action_log_probs_old: chex.Array,
        type: str = 'regular'
    ) -> chex.Array:
        mu, sigma = self.policy_network.apply(policy_params, observation)
        action_log_probs = rlax.gaussian_diagonal().logprob(action, mu, sigma)

        if type == 'clipped':
            ratio = self.prob_ratio(policy_params, observation, action, action_log_probs_old)
            normalized_advantage  = self.normalize_advantage(advantage)
            clipped_ratio = jnp.clip(ratio, a_min=1-self.eps, a_max=1+self.eps)
            loss = - jnp.mean(jnp.minimum(ratio*normalized_advantage, clipped_ratio*normalized_advantage))
        else : 
            loss = -jnp.mean(action_log_probs * advantage)
        return loss 
   