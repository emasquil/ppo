import chex


@chex.dataclass
class Transition:
    observation_t: chex.ArrayNumpy
    done_t: bool
    action_t: chex.ArrayNumpy
    value_t: float
    log_probability_t: float
    reward_tp1: float
    advantage_t: float
