import numpy as np

import dm_env
import chex
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

    def observe(self, value: float, log_probability: float, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        self.replay_buffer.add(value, log_probability, action, next_timestep)