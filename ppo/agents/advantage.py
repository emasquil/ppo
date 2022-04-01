import jax.numpy as jnp
import jax
import numpy as np


def general_advantage_estimation(values_t, dones_tp1, rewards_tp1, discount, gae_lambda):
    """Estimate advantage function"""
    advantages = np.zeros(len(rewards_tp1))
    lastgaelam = 0.0
    value_tp1 = values_t[-1]
    for t in reversed(range(len(rewards_tp1))):
        not_done_tp1 = 1.0 - dones_tp1[t]
        delta = rewards_tp1[t] + discount * value_tp1 * not_done_tp1 - values_t[t]
        advantages[t] = delta + discount * gae_lambda * not_done_tp1 * lastgaelam
        lastgaelam = advantages[t]
        value_tp1 = values_t[t]
    return jnp.array(advantages)


# gseneral_advantage_estimation = jax.jit(general_advantage_estimation_)
