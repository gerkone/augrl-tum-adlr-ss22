from typing import Any, Dict, Type, Callable

from d3rlpy.algos.awac import AWAC
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos .bc import BC, DiscreteBC
from d3rlpy.algos .bcq import BCQ, DiscreteBCQ
from d3rlpy.algos .bear import BEAR
from d3rlpy.algos .combo import COMBO
from d3rlpy.algos .cql import CQL, DiscreteCQL
from d3rlpy.algos .crr import CRR
from d3rlpy.algos .ddpg import DDPG
from d3rlpy.algos .dqn import DQN, DoubleDQN
from d3rlpy.algos .iql import IQL
from d3rlpy.algos .mopo import MOPO
from d3rlpy.algos .nfq import NFQ
from d3rlpy.algos .plas import PLAS, PLASWithPerturbation
from d3rlpy.algos .random_policy import DiscreteRandomPolicy, RandomPolicy
from d3rlpy.algos .sac import SAC, DiscreteSAC
from d3rlpy.algos .td3 import TD3
from d3rlpy.algos .td3_plus_bc import TD3PlusBC

from augrl.algos import AugmentedBEAR, AugmentedBCQ, AugmentedCQL, AugmentedBC
from augrl.algos import AugmentedDiscreteBCQ, AugmentedDiscreteCQL, AugmentedDiscreteBC

from d3rlpy.metrics.scorer import (
    average_value_estimation_scorer,
    continuous_action_diff_scorer,
    discounted_sum_of_advantage_scorer,
    discrete_action_match_scorer,
    dynamics_observation_prediction_error_scorer,
    dynamics_prediction_variance_scorer,
    dynamics_reward_prediction_error_scorer,
    evaluate_on_environment,
    initial_state_value_estimation_scorer,
    soft_opc_scorer,
    td_error_scorer,
    value_estimation_std_scorer,
)

DISCRETE_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "bc": DiscreteBC,
    "bcq": DiscreteBCQ,
    "cql": DiscreteCQL,
    "dqn": DQN,
    "double_dqn": DoubleDQN,
    "nfq": NFQ,
    "sac": DiscreteSAC,
    "random": DiscreteRandomPolicy,

    "AugmentedDiscreteBCQ": AugmentedDiscreteBCQ,
    "AugmentedDiscreteCQL": AugmentedDiscreteCQL,
    "AugmentedDiscreteBC": AugmentedDiscreteBC
}

CONTINUOUS_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "awac": AWAC,
    "bc": BC,
    "bcq": BCQ,
    "bear": BEAR,
    "combo": COMBO,
    "cql": CQL,
    "crr": CRR,
    "ddpg": DDPG,
    "iql": IQL,
    "mopo": MOPO,
    "plas": PLASWithPerturbation,
    "sac": SAC,
    "td3": TD3,
    "td3_plus_bc": TD3PlusBC,
    "random": RandomPolicy,

    "AugmentedBCQ": AugmentedBCQ,
    "AugmentedBEAR": AugmentedBEAR,
    "AugmentedCQL": AugmentedCQL,
    "AugmentedBC": AugmentedBC
}

SCORERS: Dict[str, Type[Callable]] = {
    "average_value_estimation_scorer": average_value_estimation_scorer,
    "continuous_action_diff_scorer": continuous_action_diff_scorer,
    "discounted_sum_of_advantage_scorer": discounted_sum_of_advantage_scorer,
    "discrete_action_match_scorer": discrete_action_match_scorer,
    "dynamics_observation_prediction_error_scorer": dynamics_observation_prediction_error_scorer,
    "dynamics_prediction_variance_scorer": dynamics_prediction_variance_scorer,
    "dynamics_reward_prediction_error_scorer": dynamics_reward_prediction_error_scorer,
    "evaluate_on_environment": evaluate_on_environment,
    "initial_state_value_estimation_scorer": initial_state_value_estimation_scorer,
    "soft_opc_scorer": soft_opc_scorer,
    "td_error_scorer": td_error_scorer,
    "value_estimation_std_scorer": value_estimation_std_scorer
}



def get_algo(name: str, discrete: bool) -> Type[AlgoBase]:
    """Returns algorithm class from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.

    Returns:
        type: algorithm class.

    """
    if discrete:
        if name in DISCRETE_ALGORITHMS:
            return DISCRETE_ALGORITHMS[name]
        raise ValueError(f"{name} does not support discrete action-space.")
    if name in CONTINUOUS_ALGORITHMS:
        return CONTINUOUS_ALGORITHMS[name]
    raise ValueError(f"{name} does not support continuous action-space.")

def get_scorers(names: list[str]) -> dict[str, Type[Callable]]:
    return {k:v for k,v in SCORERS.items() if k in names}