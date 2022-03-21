from typing import *

from acme import Actor, specs, types
import dm_env
import chex
import numpy as np


class RandomAgent(Actor):
    def __init__(self, environment_spec: specs.EnvironmentSpec) -> None:
        self.environment_spec = environment_spec

    def select_action(self, observation: types.NestedArray) -> chex.Array:
        action_shape = self.environment_spec.actions.shape
        max_action, min_action = (
            self.environment_spec.actions.maximum,
            self.environment_spec.actions.minimum,
        )
        action = np.random.uniform(min_action, max_action, size=action_shape)
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def update(self) -> Mapping[str, chex.ArrayNumpy]:
        return dict()
