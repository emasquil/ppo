import haiku as hk
from typing import Sequence, Optional
import chex
import jax

from ppo.networks.linear import LinearOrthogonal


class ValueNetwork(hk.Module):
    def __init__(
        self, hidden_layers_params: Sequence[dict], last_layer_params: dict, name: Optional[str] = None
    ) -> None:
        """
        output_sizes is the output size of each linear layers.
        """
        super().__init__(name=name)
        self._hidden_layers_params = hidden_layers_params
        self._last_layer_params = last_layer_params

    def __call__(self, x: chex.Array) -> chex.Array:
        h = x

        for idx_hidden_layer, hidden_layer_params in enumerate(self._hidden_layers_params, start=1):
            h = LinearOrthogonal(
                hidden_layer_params["output_size"],
                hidden_layer_params["std"],
                hidden_layer_params["bias"],
                f"value_layer{idx_hidden_layer}",
            )(h)
            h = jax.nn.tanh(h)

        return LinearOrthogonal(1, self._last_layer_params["std"], self._last_layer_params["bias"], "value_last_layer")(
            h
        )[..., 0]
