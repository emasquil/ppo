import jax
import jax.numpy as jnp
import dm_env
import numpy as np


class FixedReplayBuffer:
    """Fixed-size buffer to store transition tuples."""

    def __init__(self, key_replay_buffer, buffer_size, environment_spec) -> None:
        self.timestep = None
        self.last_value = None
        self._key_replay_buffer = key_replay_buffer
        self.buffer_size = buffer_size

        # # Creating arrays for storing all the transitions
        # obs_size = np.prod(environment_spec.observations.shape, dtype=int)
        # action_size = np.prod(environment_spec.actions.shape, dtype=int)
        # rewards_size = np.prod(environment_spec.rewards.shape, dtype=int)

        # # values_t stores one more step (it also stores last value)
        # self.values_t = jnp.zeros([buffer_size + 1, 1], dtype=jnp.float32)
        # self.obs_t = jnp.zeros([buffer_size, obs_size], dtype=jnp.float32)
        # self.actions_t = jnp.zeros([buffer_size, action_size], dtype=jnp.float32)
        # self.rewards_tp1 = jnp.zeros([buffer_size, rewards_size], dtype=jnp.float32)
        # self.advantages_t = jnp.zeros([buffer_size, 1], dtype=jnp.float32)
        # self.dones_tp1 = jnp.zeros([buffer_size, 1], dtype=bool)
        # self.logprobs_t = jnp.zeros([buffer_size, 1], dtype=jnp.float32)

        # # internal counter to keep track of the position inside of the buffer
        # self._i = 0

        # Creating arrays for storing all the transitions
        # obs_size = np.prod(environment_spec.observations.shape, dtype=int)
        # action_size = np.prod(environment_spec.actions.shape, dtype=int)
        # rewards_size = np.prod(environment_spec.rewards.shape, dtype=int)

        # values_t stores one more step (it also stores last value)
        self.values_t = []
        self.obs_t = []
        self.actions_t = []
        self.rewards_tp1 = []
        self.advantages_t = []
        self.dones_tp1 = []
        self.logprobs_t = []

        # internal counter to keep track of the position inside of the buffer
        self._i = 0


    def __len__(self):
        return self.buffer_size

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
        # self.values_t = self.values_t.at[self._i, :].set(value)
        # self.obs_t = self.obs_t.at[self._i, :].set(self.timestep.observation)
        # self.actions_t = self.actions_t.at[self._i, :].set(action)
        # self.rewards_tp1 = self.rewards_tp1.at[self._i, :].set(next_timestep.reward)
        # self.dones_tp1 = self.dones_tp1.at[self._i, :].set(next_timestep.last())
        # self.logprobs_t = self.obs_t.at[self._i, :].set(log_probability)

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

    def cast_to_jax(self):
        self.values_t = jnp.array(self.values_t)
        self.obs_t = jnp.array(self.obs_t)
        self.actions_t = jnp.array(self.actions_t)
        self.rewards_tp1 = jnp.array(self.rewards_tp1)
        self.dones_tp1 = jnp.array(self.dones_tp1)
        self.logprobs_t = jnp.array(self.logprobs_t)

