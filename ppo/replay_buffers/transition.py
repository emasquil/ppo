import chex


@chex.dataclass
class Transition:
    observation_t: chex.ArrayNumpy
    action_t: chex.ArrayNumpy
    value_t: float
    log_probability_t: float
    reward_tp1: float
    done_tp1: bool
    advantage_t: float
