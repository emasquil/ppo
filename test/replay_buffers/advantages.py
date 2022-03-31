import jax
import jax.numpy as jnp

from ppo.agents.advantage import general_advantage_estimation


class Case1:
    def __init__(self):
        self.values_t = jnp.array([0, 1, 2, 3], dtype=jnp.float32)
        self.rewards_tp1 = jnp.array([1, 0, 1], dtype=jnp.float32)
        self.dones_tp1 = jnp.array([0, 0, 1], dtype=bool)

    def __len__(self):
        return 3

    def correct_gae(self):
        return jnp.array([2.25, 0.5, -1])


class Case2:
    def __init__(self):
        self.values_t = jnp.array([0, 1, 2, 3], dtype=jnp.float32)
        self.rewards_tp1 = jnp.array([1, 0, 1], dtype=jnp.float32)
        self.dones_tp1 = jnp.array([0, 0, 0], dtype=bool)

    def __len__(self):
        return 3

    def correct_gae(self):
        return jnp.array([3, 2, 2])


class Case3:
    def __init__(self):
        self.values_t = jnp.array([0, 1, 2, 3], dtype=jnp.float32)
        self.rewards_tp1 = jnp.array([1, 0, 1], dtype=jnp.float32)
        self.dones_tp1 = jnp.array([0, 1, 0], dtype=bool)

    def __len__(self):
        return 3

    def correct_gae(self):
        return jnp.array([1.5, -1, 2])


if __name__ == "__main__":
    replay_buffer = Case1()
    gae = general_advantage_estimation(
        replay_buffer.values_t, replay_buffer.dones_tp1, replay_buffer.rewards_tp1, discount=1, gae_lambda=0.5
    )
    assert jnp.allclose(gae, replay_buffer.correct_gae())

    replay_buffer = Case2()
    gae = general_advantage_estimation(
        replay_buffer.values_t, replay_buffer.dones_tp1, replay_buffer.rewards_tp1, discount=1, gae_lambda=0.5
    )
    assert jnp.allclose(gae, replay_buffer.correct_gae())

    replay_buffer = Case3()
    gae = general_advantage_estimation(
        replay_buffer.values_t, replay_buffer.dones_tp1, replay_buffer.rewards_tp1, discount=1, gae_lambda=0.5
    )
    assert jnp.allclose(gae, replay_buffer.correct_gae())
