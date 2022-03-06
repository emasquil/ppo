import tree
from typing import *

from acme import Actor, specs, types
import dm_env
import chex
import numpy as np


class RandomAgent(Actor):
    def __init__(self, environment_spec: specs.EnvironmentSpec) -> None:
        self.environment_spec = environment_spec

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        batch_size = tree.flatten(observation)[0].shape[0]
        batched_actions = np.random.randn(
            batch_size, *self.environment_spec.actions.shape
        )
        return batched_actions

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def update(self) -> Mapping[str, chex.ArrayNumpy]:
        return dict()
