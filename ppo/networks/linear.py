from functools import partial

import haiku as hk
import jax.numpy as jnp


class LinearOrthogonal(hk.Module):
    
  def __init__(self, output_size, std, bias, name=None):
    super().__init__(name=name)
    self.output_size = output_size
    self.std = std
    self.bias = bias

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    
    w_init = hk.initializers.Orthogonal(self.std)
    w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    
    b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=partial(self.bias_initializer, self.bias))

    return jnp.dot(x, w) + b

  @staticmethod
  def bias_initializer(bias, shape, dtype=None):
        return bias * jnp.ones(shape, dtype=dtype)

