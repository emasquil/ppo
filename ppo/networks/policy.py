import numpy as np
import haiku as hk
from typing import Sequence, Optional, Tuple
from acme import specs
import chex
import jax
import jax.numpy as jnp


class PolicyNetwork(hk.Module):
    def __init__(self, output_sizes: Sequence[int], action_spec: specs.BoundedArray, name: Optional[str] = None) -> None:
        """
        output_sizes is the output size of each linear layers.
        """
        super().__init__(name=name)
        self._output_sizes = output_sizes
        self._action_spec = action_spec

    def __call__(self, x: chex.Array, ) -> Tuple[chex.Array, chex.Array]:
        action_shape = self._action_spec.shape
        action_dims = np.prod(action_shape)

        h = x

        for output_size in self._output_sizes:
            h = hk.Linear(output_size)(h)
            h = jax.nn.tanh(h)

        h = hk.Linear(2 * action_dims)(h)
        mu, pre_sigma = jnp.split(h, 2, axis=-1)
        sigma = jax.nn.softplus(pre_sigma)

        return hk.Reshape(action_shape)(.1 * mu), hk.Reshape(action_shape)(.1 * sigma)