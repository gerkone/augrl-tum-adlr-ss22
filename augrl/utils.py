from typing import Tuple

import numpy as np
from d3rlpy.dataset import MDPDataset


def trim(dataset: MDPDataset, ratio: float) -> MDPDataset:
    if ratio < 1:
        n_obs = len(dataset.observations)
        offset = np.random.randint(0, int(n_obs * (1 - ratio)))
        r_observations = dataset.observations[offset : int(n_obs * ratio) + offset]
        r_actions = dataset.actions[offset : int(n_obs * ratio) + offset]
        r_rewards = dataset.rewards[offset : int(n_obs * ratio) + offset]
        r_terminals = dataset.terminals[offset : int(n_obs * ratio) + offset]
        r_episode_terminals = dataset.episode_terminals[
            offset : int(n_obs * ratio) + offset
        ]
        return MDPDataset(
            r_observations, r_actions, r_rewards, r_terminals, r_episode_terminals
        )
    return dataset


def normalize(dataset: MDPDataset) -> Tuple[MDPDataset, np.ndarray, np.ndarray]:
    min_obs = dataset.observations.min(0)
    max_obs = dataset.observations.max(0)
    ptp_obs = max_obs - min_obs
    norm_observations = (dataset.observations - min_obs) / ptp_obs
    return (
        MDPDataset(
            norm_observations,
            dataset.actions,
            dataset.rewards,
            dataset.terminals,
            dataset.episode_terminals,
        ),
        min_obs,
        max_obs,
    )
