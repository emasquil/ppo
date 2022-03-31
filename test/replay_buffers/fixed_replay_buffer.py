import numpy as np
import jax

from ppo.replay_buffers import FixedReplayBuffer
from ppo.env_wrapper import PendulumEnv

if __name__ == "__main__":
    replay_buffer = FixedReplayBuffer(jax.random.PRNGKey(0))

    env = PendulumEnv()

    num_episodes = 3
    num_steps = 5
    rewards = []
    observations = []
    for e in range(num_episodes):
        timestep = env.reset()
        replay_buffer.add_first(timestep)

        for t in range(num_steps):
            action = 0.5 * np.ones((1,))
            observations.append(timestep.observation)
            timestep = env.step(action)
            replay_buffer.add(t, t, action, timestep)

            rewards.append(timestep.reward)
            if t == num_steps:
                observations.append(timestep.observation)

    observations = np.array(observations)
    rewards = np.array(rewards)
    replay_observatons = replay_buffer.obs_t
    replay_rewards = replay_buffer.rewards_tp1

    assert np.all(np.isclose(replay_observatons, observations))
    assert np.all(np.isclose(replay_rewards, rewards))

    print("Test completed")
