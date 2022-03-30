import numpy as np
import jax.numpy as jnp
import haiku as hk
import jax


from ppo.replay_buffers.base_replay_buffer import BaseReplayBuffer
from ppo.replay_buffers.transition import Transition


class DataLoader:
    def __init__(self, replay_buffer: BaseReplayBuffer, batch_size: int, key_dataloader: int) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.indexes = np.arange(0, len(self.replay_buffer))
        self._key_dataloader = key_dataloader

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
        self._key_dataloader, rng = jax.random.split(self._key_dataloader)
        self.indexes = np.array(jax.random.shuffle(rng, self.indexes))
