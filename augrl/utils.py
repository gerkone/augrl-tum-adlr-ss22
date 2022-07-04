from locale import RADIXCHAR
from subprocess import call
from typing import Dict, Union, Callable, Any, List
import functools
import torch.multiprocessing as tmp
import sys
import os
from contextlib import contextmanager

import d3rlpy.algos
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing.scalers import Scaler
from d3rlpy.metrics.scorer import AlgoProtocol
import gym
import copy 

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


def _evaluate(algo: AlgoProtocol, env: gym.Env, epsilon: float):
    observation = env.reset()
    episode_reward = 0.0
    while True:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = algo.predict([observation])[0]
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    return episode_reward


def m_evaluate_on_environment(
    env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False
) -> Callable[..., float]:

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        evaluate_fn = functools.partial(_evaluate, **{"env": env, "epsilon": epsilon})
        # hide subprocess output
        with open(os.devnull, 'w') as devnull:
            # suppress stdout
            orig_stdout_fno = os.dup(sys.stdout.fileno())
            os.dup2(devnull.fileno(), 1)
            # suppress stderr
            orig_stderr_fno = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), 2)
            
            # run multiple evaluations
            with tmp.Pool(tmp.cpu_count() - 1) as pool:
                episode_rewards: List[float] = pool.map(evaluate_fn, [copy.deepcopy(algo) for _ in range(n_trials)])
            
            # restore
            os.dup2(orig_stdout_fno, 1)
            os.dup2(orig_stderr_fno, 2)    
        return float(np.mean(episode_rewards))
    
    return scorer
