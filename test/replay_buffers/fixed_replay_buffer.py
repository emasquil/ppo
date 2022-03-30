import numpy as np
import jax

from ppo.replay_buffers import FixedReplayBuffer
from ppo.env_wrapper import PendulumEnv

if __name__ == "__main__":
    replay_buffer = FixedReplayBuffer(jax.random.PRNGKey(0))

    env = PendulumEnv()

    num_episodes = 3
    for e in range(num_episodes):
        timestep = env.reset()
        replay_buffer.add_first(timestep)

        rewards = []
        observations = []
        num_steps = 5
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
        replay_observatons = np.array([np.array(transition.observation_t) for transition in replay_buffer._memory[-1]])
        replay_rewards = np.array([np.array(transition.reward_tp1) for transition in replay_buffer._memory[-1]])

        assert np.all(np.isclose(replay_observatons, observations))
        assert np.all(np.isclose(replay_rewards, rewards))

        replay_buffer._memory.append([])
    print("Test completed")
