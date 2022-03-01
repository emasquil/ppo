import gym
import dm_env
from dm_env import specs
import numpy as np


class PendulumEnv(dm_env.Environment):
    def __init__(self) -> None:
        super().__init__()
        self._env = gym.make('Pendulum-v1')

    def reset(self) -> dm_env.TimeStep:
        """ Resets the environment and returns an initial observation.

        Note that there's no reward for this first observation

        Returns:
            dm_env.TimeStep
        """
        observation = self._env.reset()
        return dm_env.restart(observation)

    def step(self, action) -> dm_env.TimeStep:
        """Returns a new TimeStep (reward, observation) according to the given action

        Args:
            action (np array): shape=(1,) and dtype=np.float32

        Returns:
            dm_env.TimeStep
        """
        observation, reward, done, _ = self._env.step(action)
        if done:
            dm_env.termination(reward, observation)
        return dm_env.transition(reward, observation)

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the type of array used to represent observations.

        For the Pendulum-v1 environment, the observations consists of 3 values:
            - x position
            - y position
            - angular velocity: bounded in (-8, 8) 

        Returns:
            specs.BoundedArray
        """
        return specs.BoundedArray(shape=(3,), dtype=np.float32, minimum=-8.0, maximum=8.0)

    def action_spec(self) -> specs.BoundedArray: 
        """Returns the type of array used to represent actions

        For the Pendulum-v1 ennvironmet, the action is a single value representing the torque
        applied to the pendulum. It's bounded between (-2, 2)

        Returns:
            specs.BoundedArray
        """
        return specs.BoundedArray(shape=(1,), dtype=np.float32, minimum=-2.0, maximum=2.0)

    def close(self):
        self._env.close()


if __name__ == '__main__':
    
    env = gym.make('Pendulum-v1')
    obs = env.reset()

    print()

