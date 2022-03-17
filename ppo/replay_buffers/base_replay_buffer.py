import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from ppo.replay_buffers.transition import Transition


class BaseReplayBuffer:
    """Fixed-size base buffer to store transition tuples."""

    def __init__(self, buffer_capacity: int) -> None:
        self._memory = list()
        self._maxlen = buffer_capacity
        self.sampling_keys = hk.PRNGSequence(1)
    def __len__(self) -> int:
        return len(self._memory)

    def __getitem__(self, idx: int) -> Transition:
        assert 0 <= idx and idx < len(self), f"The queried index {idx} is out of scope [0, {len(self) - 1}]."
        return self._memory[idx]

    def sample_pop_a_transition(self) -> Transition:
        """Randomly sample and pop a transition from memory."""
        assert len(self._memory) > 0, "replay buffer is unfilled"

        transition_idx = jax.random.randint(next(self.sampling_keys), shape=[1], minval=0, maxval=len(self._memory))
        transition = self._memory.pop(np.array(transition_idx)[0])

        return transition

    def sample_empty_full_buffer(self) -> Transition:
        """Randomly sample and empty the enitre memory."""
        n_samples = len(self._memory)
        assert n_samples > 0, "replay buffer is unfilled"
        all_transitions = [self.sample_pop_a_transition() for _ in range(n_samples)] 

        stacked_transitions = {}
        for attribute in all_transitions[0]:
          arrays = [transition[attribute] for transition in all_transitions]
          arrays = jnp.stack(arrays, axis=0)
          stacked_transitions[attribute] = arrays

        return Transition(**stacked_transitions)

    def clear(self) -> None:
        """Empty the entire buffer"""
        self._memory = list()