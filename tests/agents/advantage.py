import numpy as np

from ppo.replay_buffers import Transition
from ppo.agents.advantage import n_step_bootstrap


def get_transition(value_t, reward_tp1, done_tp1):
    return Transition(
        observation_t=None,
        action_t=None,
        value_t=value_t,
        log_probability_t=None,
        reward_tp1=reward_tp1,
        done_tp1=done_tp1,
        advantage_t=None,
    )


class TestNStepBootstrap:
    def test_one_transition_done_true(self):
        discount = np.random.random()
        value_0 = np.random.random() * 100
        reward_1 = np.random.random() * 100
        value_1 = np.random.random() * 100
        done_1 = True

        one_transition = [get_transition(value_0, reward_1, done_1)]

        advantage = n_step_bootstrap(one_transition, value_1, discount)
        one_step_bootstrap = reward_1 - value_0

        assert len(advantage) == 1
        np.testing.assert_almost_equal(
            advantage[0], one_step_bootstrap, decimal=5, err_msg=f"{advantage[0]} != {one_step_bootstrap}"
        )

    def test_one_transition_done_false(self):
        discount = np.random.random()
        value_0 = np.random.random() * 100
        reward_1 = np.random.random() * 100
        value_1 = np.random.random() * 100
        done_1 = False

        one_transition = [get_transition(value_0, reward_1, done_1)]

        advantage = n_step_bootstrap(one_transition, value_1, discount)
        one_step_bootstrap = reward_1 + discount * value_1 - value_0

        assert len(advantage) == 1
        np.testing.assert_almost_equal(
            advantage[0], one_step_bootstrap, decimal=5, err_msg=f"{advantage[0]} != {one_step_bootstrap}"
        )

    def test_two_transitions_done_false_true(self):
        discount = np.random.random()
        value_0 = np.random.random() * 100
        reward_1 = np.random.random() * 100
        value_1 = np.random.random() * 100
        done_1 = False
        reward_2 = np.random.random() * 100
        value_2 = np.random.random() * 100
        done_2 = True

        two_transitions = [get_transition(value_0, reward_1, done_1), get_transition(value_1, reward_2, done_2)]

        advantage = n_step_bootstrap(two_transitions, value_2, discount)
        two_step_bootstrap_transition_1 = (
            reward_1 + discount * reward_2 - value_0
        )
        two_step_bootstrap_transition_2 = reward_2 - value_1

        assert len(advantage) == 2
        np.testing.assert_almost_equal(
            advantage[0],
            two_step_bootstrap_transition_1,
            decimal=5,
            err_msg=f"{advantage[0]} != {two_step_bootstrap_transition_1}",
        )
        np.testing.assert_almost_equal(
            advantage[1],
            two_step_bootstrap_transition_2,
            decimal=5,
            err_msg=f"{advantage[1]} != {two_step_bootstrap_transition_2}",
        )


    def test_two_transitions_done_false_false(self):
        discount = np.random.random()
        value_0 = np.random.random() * 100
        reward_1 = np.random.random() * 100
        value_1 = np.random.random() * 100
        done_1 = False
        reward_2 = np.random.random() * 100
        value_2 = np.random.random() * 100
        done_2 = False

        two_transitions = [get_transition(value_0, reward_1, done_1), get_transition(value_1, reward_2, done_2)]

        advantage = n_step_bootstrap(two_transitions, value_2, discount)
        two_step_bootstrap_transition_1 = (
            reward_1 + discount * reward_2 + discount ** 2 * value_2 - value_0
        )
        two_step_bootstrap_transition_2 = reward_2 + discount * value_2 - value_1

        assert len(advantage) == 2
        np.testing.assert_almost_equal(
            advantage[0],
            two_step_bootstrap_transition_1,
            decimal=5,
            err_msg=f"{advantage[0]} != {two_step_bootstrap_transition_1}",
        )
        np.testing.assert_almost_equal(
            advantage[1],
            two_step_bootstrap_transition_2,
            decimal=5,
            err_msg=f"{advantage[1]} != {two_step_bootstrap_transition_2}",
        )

    def test_two_transitions_done_true_false(self):
        discount = np.random.random()
        value_0 = np.random.random() * 100
        reward_1 = np.random.random() * 100
        value_1 = np.random.random() * 100
        done_1 = True
        reward_2 = np.random.random() * 100
        value_2 = np.random.random() * 100
        done_2 = False

        two_transitions = [get_transition(value_0, reward_1, done_1), get_transition(value_1, reward_2, done_2)]

        advantage = n_step_bootstrap(two_transitions, value_2, discount)
        two_step_bootstrap_transition_1 = (
            reward_1 - value_0
        )
        two_step_bootstrap_transition_2 = reward_2 + discount * value_2 - value_1

        assert len(advantage) == 2
        np.testing.assert_almost_equal(
            advantage[0],
            two_step_bootstrap_transition_1,
            decimal=5,
            err_msg=f"{advantage[0]} != {two_step_bootstrap_transition_1}",
        )
        np.testing.assert_almost_equal(
            advantage[1],
            two_step_bootstrap_transition_2,
            decimal=5,
            err_msg=f"{advantage[1]} != {two_step_bootstrap_transition_2}",
        )
