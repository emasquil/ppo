import numpy as np

import jax


class BaseReplayBuffer:
    """Fixed-size base buffer to store transition tuples."""

    def __init__(self, buffer_capacity: int):
        """Initialize a ReplayBuffer object.
        Args:
            batch_size (int): size of each training batch
        """
        self._memory = list()
        self._maxlen = buffer_capacity

    def sample_a_transition(self):
        """Randomly sample a transition from memory."""
        assert len(self._memory) > 0, "replay buffer is unfilled"

        transition_idx = np.random.randint(0, len(self._memory))
        transition = self._memory.pop(transition_idx)

        return transition

    def sample_full_buffer(self):
        """Randomly sample a transition from memory."""
        assert len(self._memory) > 0, "replay buffer is unfilled"

        transition_idxs = np.arange(0, len(self._memory))
        np.random.shuffle(transition_idxs)
        transitions = jax.tree_util.tree_map(
            lambda transition_idx: self._memory.pop(transition_idx), transition_idxs
        )

        assert len(self._memory) == 0, "replay buffer has to be empty after sampling it"

        return transitions
