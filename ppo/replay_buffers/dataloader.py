import numpy as np
import jax.numpy as jnp
import haiku as hk
import jax


from ppo.replay_buffers.base_replay_buffer import BaseReplayBuffer
from ppo.replay_buffers.transition import Transition


class DataLoader:
    def __init__(self, replay_buffer: BaseReplayBuffer, batch_size: int, key_shuffling_batch: int) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.indexes = np.arange(0, len(self.replay_buffer))
        self.suffling_keys = hk.PRNGSequence(key_shuffling_batch)

    def __len__(self) -> int:
        return np.ceil(len(self.replay_buffer) / self.batch_size).astype(int)

    def __getitem__(self, idx: int) -> Transition:
        assert 0 <= idx and idx <= len(self), f"The queried index {idx} is out of scope [0, {len(self)}]."
        if idx == len(self):
            raise StopIteration

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.replay_buffer))

        all_transitions = [self.replay_buffer[self.indexes[idx_sample]] for idx_sample in range(start, end)] 

        stacked_transitions = {}
        for attribute in all_transitions[0]:
          arrays = [transition[attribute] for transition in all_transitions]
          arrays = jnp.stack(arrays, axis=0)
          stacked_transitions[attribute] = arrays

        return Transition(**stacked_transitions)

    def shuffle(self) -> None:
        self.indexes = np.array(jax.random.shuffle(next(self.suffling_keys), self.indexes))

    def get_full_memory(self) -> list:
        return self.replay_buffer._memory