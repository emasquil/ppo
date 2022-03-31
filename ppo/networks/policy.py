import numpy as np
import haiku as hk
from typing import Sequence, Optional, Tuple
from acme import specs
import chex
import jax
import jax.numpy as jnp

from ppo.networks.linear import LinearOrthogonal


class PolicyNetFixedSigma(hk.Module):
    def __init__(
        self,
        hidden_layers_params: Sequence[dict],
        last_layer_params: dict,
        action_spec: specs.BoundedArray,
        name: Optional[str] = None,
        sigma: float = 0.5,
    ) -> None:
        """
        output_sizes is the output size of each linear layers.
        """
        super().__init__(name=name)
        self._hidden_layers_params = hidden_layers_params
        self._last_layer_params = last_layer_params
        self._action_spec = action_spec
        self._sigma = sigma

    def __call__(
        self,
        x: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        action_shape = self._action_spec.shape
        action_dims = np.prod(action_shape)

        h = x

        for idx_hidden_layer, hidden_layer_params in enumerate(self._hidden_layers_params, start=1):
            h = LinearOrthogonal(
                hidden_layer_params["output_size"],
                hidden_layer_params["std"],
                hidden_layer_params["bias"],
                f"policy_layer{idx_hidden_layer}",
            )(h)
            h = jax.nn.tanh(h)

        h = LinearOrthogonal(
            action_dims, self._last_layer_params["std"], self._last_layer_params["bias"], "policy_last_layer"
        )(h)

        h = jax.nn.tanh(h)

        return hk.Reshape(action_shape)(h), self._sigma * jnp.ones(action_dims)


class PolicyNetComplete(hk.Module):
    def __init__(
        self,
        hidden_layers_params: Sequence[dict],
        last_layer_params: dict,
        action_spec: specs.BoundedArray,
        min_sigma: float,
        init_sigma: float,
        name: Optional[str] = None,
    ) -> None:
        """
        output_sizes is the output size of each linear layers.
        """
        super().__init__(name=name)
        self._hidden_layers_params = hidden_layers_params
        self._last_layer_params = last_layer_params
        self._action_spec = action_spec
        self._min_sigma = min_sigma
        self._init_sigma = init_sigma

    def __call__(
        self,
        x: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        action_shape = self._action_spec.shape
        action_dims = np.prod(action_shape)

        h = x

        for idx_hidden_layer, hidden_layer_params in enumerate(self._hidden_layers_params, start=1):
            h = LinearOrthogonal(
                hidden_layer_params["output_size"],
                hidden_layer_params["std"],
                hidden_layer_params["bias"],
                f"policy_layer{idx_hidden_layer}",
            )(h)
            h = jax.nn.relu(h)

        h = LinearOrthogonal(
            2 * action_dims, self._last_layer_params["std"], self._last_layer_params["bias"], "policy_last_layer"
        )(h)
        h = jax.nn.tanh(h)

        mu, pre_sigma = jnp.split(h, 2, axis=-1)
        sigma = jax.nn.softplus(pre_sigma)
        sigma *= self._init_sigma / jax.nn.softplus(0.0)
        sigma += self._min_sigma
        return hk.Reshape(action_shape)(mu * 0.1), hk.Reshape(action_shape)(sigma * 0.1)
