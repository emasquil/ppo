from typing import Dict

import numpy as np
import jax.numpy as jnp
import haiku as hk
import jax


from ppo.replay_buffers import FixedReplayBuffer
from time import time


class DataLoader:
    def __init__(self, replay_buffer: FixedReplayBuffer, batch_size: int, key_dataloader: int) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.indexes = np.arange(0, len(self.replay_buffer.dones_tp1))
        self._key_dataloader = key_dataloader

    def __len__(self) -> int:
        return np.ceil(len(self.replay_buffer) / self.batch_size).astype(int)

    def __getitem__(self, idx: int) -> Dict:
        assert 0 <= idx and idx <= len(self), f"The queried index {idx} is out of scope [0, {len(self)}]."
        if idx == len(self):
            raise StopIteration

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.replay_buffer))
        idxs = self.indexes[start:end]

        transitions = {
            "observation_t": jnp.array(self.replay_buffer.obs_t[idxs]),
            "advantage_t": jnp.expand_dims(jnp.array(self.replay_buffer.advantages_t[idxs]), 1),
            "action_t": jnp.array(self.replay_buffer.actions_t[idxs]),
            "log_probability_t": jnp.array(self.replay_buffer.logprobs_t[idxs]),
            "value_t": jnp.array(self.replay_buffer.values_t[idxs]),
        }

        return transitions

    def shuffle(self) -> None:
        self._key_dataloader, rng = jax.random.split(self._key_dataloader)
        self.indexes = np.array(jax.random.permutation(rng, self.indexes, independent=True))
