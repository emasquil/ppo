from time import time
import numpy as np

import dm_env
from acme import Actor
import haiku as hk
import jax.numpy as jnp
from jax import tree_util
import jax
import rlax
import chex
from acme import specs
import optax


class OnlinePPO(Actor):
    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        network,
        rng: chex.PRNGKey,
        epsilon: float=0.0,
        learning_rate: float=1e-3,
        discount: float=0.99,
    ):
        self.network = hk.without_apply_rng(hk.transform(network))
        self.network_params = self.network.init(rng=rng, inputs=jnp.zeros(environment_spec.observations.shape))
        
        self.optimizer = optax.adam(learning_rate)
        self.optimizer_state = self.optimizer.init(self.network_params)
        
        def greedy_policy(network_params, rng, observation):
            action_values = self.network.apply(network_params, observation)

            return rlax.epsilon_greedy(epsilon).sample(rng, action_values)

        self.greedy_policy = greedy_policy
        self.greedy_rng = hk.PRNGSequence(1)

        self.timestep = None
        self.action = None
        self.next_timestep = None
        self.discount = discount

        # # Create a learner that updates the parameters (and initializes them).
        # self._learner = DQNLearner(
        #     network=network,
        #     obs_spec=environment_spec.observations,
        #     rng=hk.PRNGSequence(1),
        #     optimizer=optax.adam(learning_rate),
        #     discount=discount,
        # )

        # # We'll ignore the first min_observations when determining whether to take
        # # a step and we'll do so by making sure num_observations >= 0.
        # self._num_observations = -max(batch_size, min_replay_size)

        # observations_per_step = float(batch_size) / samples_per_insert
        # if observations_per_step >= 1.0:
        #     self._observations_per_update = int(observations_per_step)
        #     self._learning_steps_per_update = 1
        # else:
        #     self._observations_per_update = 1
        #     self._learning_steps_per_update = int(1.0 / observations_per_step)

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        # Convert to get a batch shape
        observation = tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), observation)

        action = self.greedy_policy(self.network_params, next(self.greedy_rng), observation)

        # Convert back to single action
        action = tree_util.tree_map(lambda x: jnp.array(x).squeeze(axis=0), action)

        return action

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        self.timestep = timestep
        self.action = None 
        self.next_timestep = None

    def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        assert self.timestep is None, "Please let the agent observe a first timestep."
        self.action = action

        if self.timestep.first():
            self.next_timestep = next_timestep
        else:
            self.timestep = self.next_timestep
            self.next_timestep = next_timestep

    def loss(self, network_params: hk.Params, timestep: dm_env.TimeStep, action: np.ndarray, next_timestep: dm_env.TimeStep) -> chex.Array:
        q_next_state = self.network.apply(network_params, next_timestep.observation)
        q_next_state = jax.lax.stop_gradient(q_next_state)

        # compute the target y_t
        y_t = timestep.reward + self.discount * jnp.max(q_next_state, axis=-1)

        ### Compute the estimate qa_tm1
        # Estimate q_tm1 with the online network
        q_tm1 = self.network.apply(params, o_tm1) 

        # Perform batch indexing to obtain q_tm1(ob_tm1, a_tm1)
        qa_tm1 = jax.vmap(lambda q, a: q[a])(q_tm1, a_tm1)

        ### Compute the final loss
        # Compute the final loss
        td_error = y_t - qa_tm1

        # Compute the L2 error in expectation
        q_loss = 0.5 * jnp.square(td_error)
        q_loss = jnp.mean(q_loss)

        return q_loss
        
    def update(self) -> None:
        assert self.timestep is None, "Please let the agent observe a first timestep."
        assert self.action is None or self.next_timestep is None, "Please let the agent observe a timestep."

        gradients = jax.grad(self.loss)(self.network_params, self.timestep, self.action, self.next_timestep)
    
        updates, self.optimizer_state = self.optimizer.update(gradients, self.optimizer_state)
        self.network_params = optax.apply_updates(self.network_params, updates)

        ## Update twice if last step
