from typing import Dict, Union

import d3rlpy.algos
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing.scalers import Scaler

import augrl.algos


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


def get_scaling_factor(scaler: Scaler) -> Union[np.ndarray, float]:
    if scaler is None:
        return 1.0
    if scaler.get_type() == "min_max":
        params = scaler.get_params()
        return params["maximum"] - params["minimum"]
    if scaler.get_type() == "standard":
        params = scaler.get_params()
        return params["std"]
    return 1.0


def merge_dicts(d1: Dict, d2: Dict) -> Dict:
    return {**d1, **d2}


def get_algo(name: str, discrete: bool) -> d3rlpy.algos.AlgoBase:
    try:
        return d3rlpy.algos.get_algo(name, discrete)
    except ValueError:
        return augrl.algos.get_algo(name, discrete)
