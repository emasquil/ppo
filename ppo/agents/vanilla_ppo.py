import numpy as np
from dataclasses import replace

import dm_env
import jax.numpy as jnp
import chex
from acme import specs

from ppo.agents.base_agent import BaseAgent
from ppo.replay_buffers import FixedReplayBuffer
from ppo.replay_buffers.transition import Transition


class VanillaPPO(BaseAgent):
    """Still need to code the methods:
    value_loss
    policy_loss
    update
    """

    def __init__(
        self,
        observation_spec: specs.BoundedArray,
        policy_network,
        value_network,
        key: chex.PRNGKey,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
    ):
        super(VanillaPPO, self).__init__(observation_spec, policy_network, value_network, key, learning_rate, discount)
        self.replay_buffer = FixedReplayBuffer(buffer_capacity=50)

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        self.replay_buffer.add_first(timestep)

    def observe(self, value: float, log_probability: float, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        self.replay_buffer.add(value, log_probability, action, next_timestep)

    def add_advantage(self, advantages):
        self.replay_buffer.add_advantage(advantages)

    def value_loss(self, value_params, batch):
        pass

    def policy_loss(self, policy_params, batch):
        # compute ratio

        #
        pass

    def update_on_batch(self, batch: Transition):
        # Normalize batch advantage
        batch.advantage_t
        replace(batch, advantage_t=batch.advantages_t - jnp.mean(batch.advantages_t, axis=0))
        end = start + args.minibatch_size
        mb_inds = b_inds[start:end]


        _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        mb_advantages = b_advantages[mb_inds]
        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()