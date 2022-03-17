import numpy as np

import jax.numpy as jnp
import chex

from ppo.replay_buffers.transition import Transition


class BaseReplayBuffer:
    """Fixed-size base buffer to store transition tuples."""

    def __init__(self, buffer_capacity: int) -> None:
        self._memory = list()
        self._maxlen = buffer_capacity

    def sample_a_transition(self) -> Transition:
        """Randomly sample a transition from memory."""
        assert len(self._memory) > 0, "replay buffer is unfilled"

        transition_idx = np.random.randint(0, len(self._memory))
        transition = self._memory.pop(transition_idx)

        return transition

    def sample_full_buffer(self) -> chex.Array:
        """Randomly sample the full buffer."""
        n_samples = len(self._memory)
        assert n_samples > 0, "replay buffer is unfilled"
        all_transitions = [self.sample_a_transition() for _ in range(n_samples)] 

        stacked_transitions = {}
        for attribute in all_transitions[0]:
          arrays = [transition[attribute] for transition in all_transitions]
          arrays = jnp.stack(arrays, axis=0)
          stacked_transitions[attribute] = arrays

        return Transition(**stacked_transitions)
