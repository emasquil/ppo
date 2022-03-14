import jax
import jax.numpy as jnp

from ppo.replay_buffers.base_replay_buffer import BaseReplayBuffer
from ppo.replay_buffers.transition import Transition


class FixedReplayBuffer(BaseReplayBuffer):
    """Fixed-size buffer to store transition tuples."""

    def __init__(self, buffer_capacity: int):
        """Initialize a ReplayBuffer object.
        Args:
            batch_size (int): size of each training batch
        """
        super(FixedReplayBuffer, self).__init__(buffer_capacity)

    def add(
        self,
        observation_t,
        action_t,
        value_t,
        log_probability_t,
        reward_tp1,
        observation_tp1,
        done_tp1,
    ) -> None:
        """Add a new transition to memory."""
        if len(self._memory) >= self._maxlen:
            self._memory.pop(0)  # remove first elem (oldest)

        transition = Transition(
            observation_t=observation_t,
            action_t=action_t,
            value_t=value_t,
            log_probability_t=log_probability_t,
            reward_tp1=reward_tp1,
            observation_tp1=observation_tp1,
            done_tp1=done_tp1,
        )

        # convert every data into jnp array
        transition = jax.tree_util.tree_map(jnp.array, transition)

        self._memory.append(transition)
