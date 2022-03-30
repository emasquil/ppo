import numpy as np
import haiku as hk
from typing import Sequence, Optional, Tuple
from acme import specs
import chex
import jax
import jax.numpy as jnp

from ppo.networks.linear import LinearOrthogonal


class PolicyNetwork(hk.Module):
    def __init__(self, hidden_layers_params: Sequence[dict], last_layer_params: dict, action_spec: specs.BoundedArray, name: Optional[str] = None) -> None:
        """
        output_sizes is the output size of each linear layers.
        """
        super().__init__(name=name)
        self._hidden_layers_params = hidden_layers_params
        self._last_layer_params = last_layer_params
        self._action_spec = action_spec
        

    def __call__(self, x: chex.Array, ) -> Tuple[chex.Array, chex.Array]:
        action_shape = self._action_spec.shape
        action_dims = np.prod(action_shape)

        h = x

        for idx_hidden_layer, hidden_layer_params in enumerate(self._hidden_layers_params, start=1):
            h = LinearOrthogonal(hidden_layer_params["output_size"], hidden_layer_params["std"], hidden_layer_params["bias"], f"policy_layer{idx_hidden_layer}")(h)
            h = jax.nn.tanh(h)

        h = LinearOrthogonal(action_dims, self._last_layer_params["std"], self._last_layer_params["bias"], "policy_last_layer")(h)

        return hk.Reshape(action_shape)(h), 0.5 * jnp.eye(action_dims)