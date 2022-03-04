import acme
import chex
from dm_env import specs
import jax
from jax.random import PRNGKey
import numpy as np

from env_wrapper.pendulum_wrapper import PendulumEnv
from networks import PolicyNetwork




@chex.dataclass
class Trajectory:
    observations: chex.Array
    rewards: chex.Array
    dones: chex.Array


class Agent:
    def __init__(self, 
                 environment_spec: specs.EnvironmentSpec,
                 rng: int) -> None:
        self._env_spec = environment_spec


    def collect_trajectories():
        pass

    def select_action(self, 
                      observations: chex.Array):
        

        pass

    def update():
        """Peforms gradient descent update

        """
        pass 


if __name__ == '__main__':
    env = PendulumEnv()
    env_specs = acme.make_environment_spec(env)

    agent = Agent(env_specs, 
                  42
                  )

    # simple interaction loop
    max_steps = 2
    observation = env.reset()
    trajec = []
    for t in range(max_steps):
        action = np.random.uniform(-2.0, 2.0, size=(1,))
        time_step = env.step(action)
        trajec.append[time_step]
        if time_step.last():
            break

    env.close()

    print()