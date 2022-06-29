from typing import Dict

import d3rlpy.algos

from .algos import (
    AugmentedBC,
    AugmentedBCQ,
    AugmentedBEAR,
    AugmentedCQL,
    AugmentedDiscreteBC,
    AugmentedDiscreteBCQ,
    AugmentedDiscreteCQL,
)

__all__ = [
    "AugmentedBEAR",
    "AugmentedBCQ",
    "AugmentedDiscreteBCQ",
    "AugmentedCQL",
    "AugmentedDiscreteCQL",
    "AugmentedBC",
    "AugmentedDiscreteBC",
]


AUG_DISCRETE_ALGORITHMS: Dict[str, d3rlpy.algos.AlgoBase] = {
    "augmentedbcq": AugmentedDiscreteBCQ,
    "augmentedcql": AugmentedDiscreteCQL,
    "augmentedbc": AugmentedDiscreteBC,
}

AUG_CONTINUOUS_ALGORITHMS: Dict[str, d3rlpy.algos.AlgoBase] = {
    "augmentedbcq": AugmentedBCQ,
    "augmentedcqL": AugmentedCQL,
    "augmentedbc": AugmentedBC,
    "augmentedbear": AugmentedBEAR,
}


def get_algo(name: str, discrete: bool) -> d3rlpy.algos.AlgoBase:
    """Returns algorithm class from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.

    Returns:
        type: algorithm class.

    """
    if discrete:
        if name in AUG_DISCRETE_ALGORITHMS:
            return AUG_DISCRETE_ALGORITHMS[name]
        raise ValueError(f"{name} does not support discrete action-space.")
    if name in AUG_CONTINUOUS_ALGORITHMS:
        return AUG_CONTINUOUS_ALGORITHMS[name]
    raise ValueError(f"{name} does not support continuous action-space.")
