import jax.numpy as jnp


def general_advantage_estimation(trajectory, last_value, last_done, discount, gae_lambda):
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
        if t == len(trajectory) - 1:
            not_done_t = 1.0 - last_done
            value_tp1 = last_value
        else:
            not_done_t = 1.0 - trajectory[t].done_t
            value_tp1 = trajectory[t + 1].value_t
        delta = trajectory[t].reward_tp1 + discount * value_tp1 * not_done_t - trajectory[t].value_t
        advantages = advantages.at[t].set((delta + discount * gae_lambda * not_done_t * lastgaelam)[0])
        lastgaelam = advantages[t]

    return advantages
