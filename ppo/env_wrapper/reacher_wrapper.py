import gym
import dm_env
from dm_env import specs
import numpy as np
import acme

from .custom_wrappers import action_wrapper


class ReacherEnv(dm_env.Environment):
    def __init__(self) -> None:
        super().__init__()
        self._env = gym.make("Reacher-v2")
        self._env = gym.wrappers.ClipAction(self._env)
        self._env = gym.wrappers.NormalizeObservation(self._env)
        self._env = gym.wrappers.TransformObservation(self._env, lambda obs: np.clip(obs, -10, 10))
        self._env = gym.wrappers.NormalizeReward(self._env, gamma=1)
        self._env = gym.wrappers.TransformReward(self._env, lambda reward: np.clip(reward, -10, 10))
        self._env = action_wrapper.ActionNormalizer(self._env)

    def __str__(self) -> str:
        return "ReacherEnv"

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

    def observation_spec(self) -> specs.Array:
        """Returns the type of array used to represent observations.

        For the Reacher-v2 environment, the observations consists of 11 values:

        Num 	Observation 	Min 	Max 	Name (in corresponding XML file) 	Joint 	Unit
        0 	cosine of the angle of the first arm 	-Inf 	Inf 	cos(joint0) 	hinge 	unitless
        1 	cosine of the angle of the second arm 	-Inf 	Inf 	cos(joint1) 	hinge 	unitless
        2 	sine of the angle of the first arm 	-Inf 	Inf 	cos(joint0) 	hinge 	unitless
        3 	sine of the angle of the second arm 	-Inf 	Inf 	cos(joint1) 	hinge 	unitless
        4 	x-coorddinate of the target 	-Inf 	Inf 	target_x 	slide 	position (m)
        5 	y-coorddinate of the target 	-Inf 	Inf 	target_y 	slide 	position (m)
        6 	angular velocity of the first arm 	-Inf 	Inf 	joint0 	hinge 	angular velocity (rad/s)
        7 	angular velocity of the second arm 	-Inf 	Inf 	joint1 	hinge 	angular velocity (rad/s)
        8 	x-value of position_fingertip - position_target 	-Inf 	Inf 	NA 	slide 	position (m)
        9 	y-value of position_fingertip - position_target 	-Inf 	Inf 	NA 	slide 	position (m)
        10 	z-value of position_fingertip - position_target (0 since reacher is 2d and z is same for both) 	-Inf 	Inf 	NA 	slide 	position (m)
        Returns:
            specs.Array
        """
        return specs.Array(
            shape=(11,),
            dtype=np.float32,
        )

    def action_spec(self) -> specs.BoundedArray:
        """Returns the type of array used to represent actions

        For the Reacher-v2 environmet:

        Num 	Action 	Control Min 	Control Max 	Name (in corresponding XML file) 	Joint 	Unit
        0 	Torque applied at the first hinge (connecting the link to the point of fixture) 	-1 	1 	joint0 	hinge 	torque (N m)
        1 	Torque applied at the second hinge (connecting the two links) 	-1 	1 	joint1 	hinge 	torque (N m)
        Returns:
            specs.BoundedArray
        """
        return specs.BoundedArray(shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0)

    def render(self, mode):
        return self._env.render(mode)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    env = ReacherEnv()

    # single object containing all the information about our env
    env_specs = acme.make_environment_spec(env)

    # simple interaction loop
    max_steps = 50
    observation = env.reset()
    for t in range(max_steps):
        env.render("human")
        action = np.random.uniform(-1.0, 1.0, size=(2,))
        time_step = env.step(action)
        if time_step.last():
            break

    env.close()
