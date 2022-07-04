from concurrent.futures import thread
from subprocess import call
from typing import Dict, Union, Callable, Any, List

import d3rlpy.algos
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing.scalers import Scaler
from d3rlpy.metrics.scorer import AlgoProtocol,StackedObservation
import augrl.algos
import gym
import multiprocessing as mp
import copy 
from datetime import datetime

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

"""
def evaluate_on_environment(
    env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, timeout: int = 30
) -> Callable[..., float]:

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        episode_rewards = []
        
        def evaluate(env: gym.Env, algo: AlgoProtocol, epsilon: float):
            observation = env.reset()
            episode_reward = 0.0
            return 10.0
            print("thread_x")
            #start_time = datetime.now()
            while True: #and (datetime.now()-start_time).seconds < timeout:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = algo.predict([observation])[0]

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if done:
                    break
            #episode_rewards.append(episode_reward)
            return episode_reward
        
        
        with mp.Pool(mp.cpu_count()-1) as pool:
            for _ in range(n_trials):
                local_env = copy.deepcopy(env)
                ret = pool.apply_async(func=evaluate,args=(local_env,algo,epsilon))
                print(ret.get())
                episode_rewards.append(ret)
            pool.close()
            pool.join()
        
        for _ in range(n_trials):
            local_env = copy.deepcopy(env)
            thread_call(local_env,algo)
        
        print(episode_rewards)
        print(float(np.mean(episode_rewards)))
        return float(np.mean(episode_rewards))

    return scorer

"""

def evaluate_on_environment(
    env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False
) -> Callable[..., float]:

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        futures = []
        episode_rewards = []
        with mp.Pool(mp.cpu_count()-1) as pool:
            for _ in range(n_trials):
                local_env = copy.deepcopy(env)
                reward = pool.apply_async(func=evaluate,args=(local_env, algo, epsilon))
                futures.append(reward)
            pool.close()
            pool.join()
        episode_rewards = [item.get() for item in futures]
        print(episode_rewards)
            
        return float(np.mean(episode_rewards))

    return scorer

def evaluate(env: gym.Env, algo: AlgoProtocol, epsilon: float):
    observation = env.reset()
    episode_reward = 0.0
    while True:
        # take action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = algo.predict([observation])[0]
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    return episode_reward