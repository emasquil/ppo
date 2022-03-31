from typing import Sequence

import numpy as np
import jax.numpy as jnp
from ppo.replay_buffers.transition import Transition


def general_advantage_estimation(trajectory, last_value, discount, gae_lambda):
    """Estimate advantage function

    Args:
        trajectory (_type_): _description_
        agent (_type_): _description_
        last_timestep (_type_): _description_
        discount (float, optional): _description_.
        gae_lambda (float, optional): _description_.

    Returns:
        _type_: _description_
    """
    advantages = jnp.zeros(len(trajectory))
    lastgaelam = 0.0
    for t in reversed(range(len(trajectory))):
        not_done_tp1 = 1.0 - trajectory[t].done_tp1
        if t == len(trajectory) - 1:
            value_tp1 = last_value
        else:
            value_tp1 = trajectory[t + 1].value_t
        delta = trajectory[t].reward_tp1 + discount * value_tp1 * not_done_tp1 - trajectory[t].value_t
        advantages = advantages.at[t].set((delta + discount * gae_lambda * not_done_tp1 * lastgaelam)[0])
        lastgaelam = advantages[t]

    return advantages


def n_step_bootstrap(trajectory: Sequence[Transition], last_value: float, discount: float) -> jnp.ndarray:
    # n-step_t = r_{t+1} + gamma r_{t+2} + gamma^2 r_{t+3} + ... + gamma^n V_{t+n}
    # if t + 1 is the last step: n-step_t = r_{t+1}
    # t can't be the last step 

    trajectory_length = len(trajectory)
    bootstrapped_value = np.zeros(trajectory_length)
    values = np.zeros(trajectory_length)

    bootstrapped_value[-1] = trajectory[-1].reward_tp1 + discount * (1 - float(trajectory[-1].done_tp1))  * last_value

    for t in reversed(range(trajectory_length - 1)):
        bootstrapped_value[t] = trajectory[t].reward_tp1 + discount * (1 - float(trajectory[t].done_tp1)) * bootstrapped_value[t + 1]
    
    for t in range(trajectory_length):
        values[t] = trajectory[t].value_t

    return jnp.array(bootstrapped_value - values)