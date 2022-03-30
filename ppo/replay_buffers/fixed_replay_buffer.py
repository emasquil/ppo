import jax
import jax.numpy as jnp
import dm_env
import numpy as np

from ppo.replay_buffers.base_replay_buffer import BaseReplayBuffer
from ppo.replay_buffers.transition import Transition


class FixedReplayBuffer(BaseReplayBuffer):
    """Fixed-size buffer to store transition tuples."""

    def __init__(self, key_replay_buffer) -> None:
        super(FixedReplayBuffer, self).__init__(key_replay_buffer)
        self.timestep = None
        self.action = None
        self.next_timestep = None
        self.last_value_and_done = (None, None)

    def add_first(self, timestep: dm_env.TimeStep) -> None:
        # Create a new trajectory for the new episode
        self._memory.append([])
        self.timestep = timestep
        self.action = None
        self.next_timestep = None

    def add(self, value: float, log_probability: float, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        """Add a new transition to memory."""
        assert self.timestep is not None, "Please let the agent observe a first timestep."
        # Convert to get a batch shape
        self.action = action

        if self.timestep.first():
            self.next_timestep = next_timestep
        else:
            self.timestep = self.next_timestep
            self.next_timestep = next_timestep

        transition = Transition(
            observation_t=self.timestep.observation,
            action_t=action,
            value_t=value,
            log_probability_t=log_probability,
            reward_tp1=self.next_timestep.reward,
            done_tp1=self.next_timestep.last(),
            advantage_t=None,
        )

        # convert every data into jnp array
        transition = jax.tree_util.tree_map(jnp.array, transition)

        # we add the transition to the current episode list of transitions
        self._memory[-1].append(transition)
