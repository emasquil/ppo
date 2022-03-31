import jax
import jax.numpy as jnp
import dm_env
import numpy as np


class FixedReplayBuffer:
    """Fixed-size buffer to store transition tuples."""

    def __init__(self, key_replay_buffer) -> None:
        self.timestep = None
        self.last_value = None
        self._key_replay_buffer = key_replay_buffer

        # values_t stores one more step (it also stores last value)
        self.values_t = []
        self.obs_t = []
        self.actions_t = []
        self.rewards_tp1 = []
        self.advantages_t = []
        self.dones_tp1 = []
        self.logprobs_t = []

    def __len__(self):
        return len(self.dones_tp1)

    def add_first(self, timestep: dm_env.TimeStep) -> None:
        self.timestep = timestep

    def add(self, value: float, log_probability: float, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        """Add a new transition to memory."""
        assert self.timestep is not None, "Please let the agent observe a first timestep."

        self.values_t.append(value)
        self.obs_t.append(self.timestep.observation)
        self.actions_t.append(action)
        self.rewards_tp1.append(next_timestep.reward)
        self.dones_tp1.append(next_timestep.last())
        self.logprobs_t.append(log_probability)
        self.timestep = next_timestep

    def add_last_value(self, value: float) -> None:
        self.values_t.append(value)

    def clear_memory(self):
        self.values_t = []
        self.obs_t = []
        self.actions_t = []
        self.rewards_tp1 = []
        self.advantages_t = []
        self.dones_tp1 = []
        self.logprobs_t = []

    def add_advantages(self, advantages):
        self.advantages_t = advantages

    def cast_to_numpy(self):
        self.values_t = np.array(self.values_t)
        self.obs_t = np.array(self.obs_t)
        self.actions_t = np.array(self.actions_t)
        self.rewards_tp1 = np.array(self.rewards_tp1)
        self.dones_tp1 = np.array(self.dones_tp1)
        self.logprobs_t = np.array(self.logprobs_t)
