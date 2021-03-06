import gym
import dm_env
from dm_env import specs
import numpy as np
import acme

from .custom_wrappers import action_wrapper


class PendulumEnv(dm_env.Environment):
    def __init__(self) -> None:
        super().__init__()
        self._env = gym.make("Pendulum-v1")
        self._env = gym.wrappers.ClipAction(self._env)
        self._env = gym.wrappers.NormalizeObservation(self._env)
        self._env = gym.wrappers.TransformObservation(self._env, lambda obs: np.clip(obs, -10, 10))
        self._env = gym.wrappers.NormalizeReward(self._env, gamma=1)
        self._env = gym.wrappers.TransformReward(self._env, lambda reward: np.clip(reward, -10, 10))
        self._env = action_wrapper.ActionNormalizer(self._env)

    def __str__(self) -> str:
        return "PendulumEnv"

    def reset(self) -> dm_env.TimeStep:
        """Resets the environment and returns an initial observation.

        Note that there's no reward for this first observation

        Returns:
            dm_env.TimeStep
        """
        self._env.returns[0] = 0.0
        observation = self._env.reset()
        return dm_env.restart(observation)

    def step(self, action) -> dm_env.TimeStep:
        """Returns a new TimeStep (reward, observation) according to the given action

        Args:
            action: castable to np.array

        Returns:
            dm_env.TimeStep:
        """
        # if action is not cast to numpy, we observe weird behaviour when taking steps
        observation, reward, done, _ = self._env.step(np.array(action))
        if done:
            return dm_env.termination(reward, observation)
        return dm_env.transition(reward, observation)

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the type of array used to represent observations.

        For the Pendulum-v1 environment, the observations consists of 3 values:
            - x position: bounded in (-1, 1)
            - y position: bounded in (-1, 1)
            - angular velocity: bounded in (-8, 8)

        Returns:
            specs.BoundedArray
        """
        return specs.BoundedArray(
            shape=(3,),
            dtype=np.float32,
            minimum=[-1.0, -1.0, -8.0],
            maximum=[1.0, 1.0, 8.0],
        )

    def action_spec(self) -> specs.BoundedArray:
        """Returns the type of array used to represent actions

        For the Pendulum-v1 environmet, the action is a single value representing the torque
        applied to the pendulum. It's bounded between (-2, 2)

        Returns:
            specs.BoundedArray
        """
        return specs.BoundedArray(shape=(1,), dtype=np.float32, minimum=-2.0, maximum=2.0)

    def render(self, mode):
        return self._env.render(mode)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    env = PendulumEnv()

    # single object containing all the information about our env
    env_specs = acme.make_environment_spec(env)

    # simple interaction loop
    max_steps = 300
    observation = env.reset()
    for t in range(max_steps):
        env.render("human")
        action = np.random.uniform(-2.0, 2.0, size=(1,))
        time_step = env.step(action)
        if time_step.last():
            break

    env.close()
