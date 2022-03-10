import haiku as hk
from typing import Sequence, Optional
import chex
import jax


class ValueNetwork(hk.Module):
    def __init__(self, output_sizes: Sequence[int], name: Optional[str] = None) -> None:
        """
        output_sizes is the output size of each linear layers.
        """
        super().__init__(name=name)
        self._output_sizes = output_sizes

    def __call__(self, x: chex.Array) -> chex.Array:
        h = x

        for output_size in self._output_sizes:
            h = hk.Linear(output_size)(h)
            h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
            h = jax.nn.relu(h)

        return hk.Linear(1)(h)[..., 0]