from typing import Dict, List, Tuple, Type, Callable
import datetime
import os
import sys

import d4rl.offline_env
import argparse
import numpy as np
import random
import yaml
import traceback
import d3rlpy
from d3rlpy.datasets import MDPDataset
import gym
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from urllib import request
from augrl import utils
from sklearn.model_selection import train_test_split

env_dict = {
    'cartpole-replay': {
        'data_path': "cartpole_replay_v1.1.0.h5",
        'url': "https://www.dropbox.com/s/uep0lzlhxpi79pd/cartpole_v1.1.0.h5?dl=1",
        'env': "CartPole-v1",
        'discrete': True
    },
    'cartpole-random': {
        'data_path': "cartpole_random_v1.1.0.h5",
        'url': "https://www.dropbox.com/s/4lgai7tgj84cbov/cartpole_random_v1.1.0.h5?dl=1",
        'env': "CartPole-v1",
        'discrete': True
    },
    'pendulum-replay': {
        'data_path': "pendulum_replay_v1.1.0.h5",
        'url': "https://www.dropbox.com/s/ukkucouzys0jkfs/pendulum_v1.1.0.h5?dl=1",
        'env': "Pendulum-v1",
        'discrete': False
    },
    'pendulum-random': {
        'data_path': "pendulum_random_v1.1.0.h5",
        'url': "https://www.dropbox.com/s/hhbq9i6ako24kzz/pendulum_random_v1.1.0.h5?dl=1",
        'env': "Pendulum-v1",
        'discrete': False
    },

}

def get_dataset(env_name: str) -> Tuple[MDPDataset, gym.Env, Dict, bool, int]:
    if ("cartpole" in env_name or "pendulum" in env_name):
        data_path = os.path.join("datasets", env_dict[env_name]["data_path"])
        if not os.path.exists(data_path):
            os.makedirs("datasets", exist_ok=True)
            request.urlretrieve(env_dict[env_name]["url"], data_path)
        env = gym.make(env_dict[env_name]["env"])
        discrete = env_dict[env_name]["discrete"]
        dataset = MDPDataset.load(data_path)
        #TODO: cartpole is discrete, however when dataset is loaded, MDPDataset sets the discrete flag to false
        # temporary workaround
        dataset = MDPDataset(
            observations = dataset.observations,
            actions = dataset.actions,
            rewards = dataset.rewards,
            terminals = dataset.terminals,
            episode_terminals = dataset.episode_terminals,
            discrete_action = discrete
        )

    else:
        env = gym.make(env_name)
        dataset = env.get_dataset()
        discrete = False

        dataset = MDPDataset(
            observations=np.array(dataset["observations"], dtype=np.float32),
            actions=np.array(dataset["actions"], dtype=np.float32),
            rewards=np.array(dataset["rewards"], dtype=np.float32),
            terminals=np.array(dataset["terminals"], dtype=np.float32),
            episode_terminals=np.array(np.logical_or(dataset["terminals"], dataset["timeouts"]), dtype=np.float32),
            discrete_action=discrete
        )

    limits = {}
    limits["obs_min"] = env.observation_space.low
    limits["obs_max"] = env.observation_space.high
    if not discrete:
        limits["action_min"] = env.action_space.low
        limits["action_max"] = env.action_space.high

    size = len(dataset.observations)

    return dataset, env, limits, discrete, size

def get_ratio_dataset(dataset: MDPDataset, ratio: float, size: int, is_discrete: bool) -> MDPDataset:
    indices = random.sample(range(size), int(size * ratio))
    return MDPDataset(
            observations = np.array([dataset.observations[index] for index in indices]),
            actions = np.array([dataset.actions[index] for index in indices]),
            rewards = np.array([dataset.rewards[index] for index in indices]),
            terminals = np.array([dataset.terminals[index] for index in indices]),
            episode_terminals = np.array([dataset.episode_terminals[index] for index in indices]), #(np.random.rand(int(size * ratio)) > 0.95).astype(int),
            discrete_action = is_discrete
    )

def train(full_dataset: MDPDataset, data_ratio: float, env: gym.Env, env_limits: Dict, is_discrete: bool, ds_size: int, algo: Callable, algo_args: Dict, config: Dict, scorers: Dict, seed: int):
    #get restricted dataset
    dataset = get_ratio_dataset(full_dataset, data_ratio, ds_size, is_discrete)
    #train-test split
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    from ipywidgets import interact, widgets
    from IPython.display import display

    rdbuttons = widgets.RadioButtons(
        description='Choose dataset:',
        layout={'width': 'max_content'},
        disabled=False,
    )

    #instantiate algorithm
    agent = algo(
        use_gpu=config["cuda"],
        augmentations=config["augmentations"],
        real_ratio=config["real_ratio"],
        scaler=config["scaler"],
        **algo_args,
    )
    agent.generated_maxlen = config.get("generated_maxlen", 100000)
    agent.limits = env_limits
    n_steps = int(ds_size * data_ratio * 3)
    #start training
    metrics = agent.fit(
        dataset=train_episodes,
        eval_episodes=test_episodes,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps // config.get("epochs", 50),  # to have 100 epochs
        scorers=scorers,
        shuffle=True,
        verbose=config["verbose"],
        show_progress=config["show_progress"],
        save_metrics=True,
        save_interval=config.get("epochs", 50),
        with_timestamp=False,
        logdir="d3rlpy_logs/{}_{}_{}".format(
            algo.__name__, env.unwrapped.spec.id, seed
        ),
        experiment_name="{}_{}_{}".format(
            data_ratio,
            env.unwrapped.spec.id,
            algo.__name__,
        ),
    )
    return metrics


def save_results(results: List, scorers: List, path: str):
    results_df = pd.DataFrame(results)
    results_df.insert(0, 'date-time', datetime.datetime.now().strftime("%d%m%Y_%H%M"))
    print(results_df.columns.tolist())
    frames = []
    for score in scorers:
        partial_df = results_df[["date-time", "env", "algo", "epoch", score]].dropna()
        partial_df = partial_df.rename({score: "value"}, axis=1)
        partial_df["metric"] = score
        frames.append(partial_df)

    results_df = pd.concat(frames)


    if (os.path.exists(path)):
        print("File already exists - appending new data")
        results_df = pd.concat([pd.read_parquet(path), results_df])

    results_df.to_parquet(path)


def run(config: Dict, seed: int) -> List:
    d4rl.set_dataset_path(os.path.join(os.getcwd(), "datasets"))
    thrown_errors = []
    d3rlpy.seed(seed)

    for env_item in config["environments"]:
        results = [] # initialize results for the environment
        try:
            full_dataset, env, env_limits, is_discrete, ds_size = get_dataset(env_item)
        except NotImplementedError:
            continue
        print(f"INFO:  Loaded {env_item}: discrete={is_discrete}, #observation={ds_size}")
        env.seed(seed = seed)

        #TODO: IF using BC, scorers are not usable except env_on_environment
        scorers = utils.get_scorers(
            scorer_names=config.get("scorers", []),
            env=env,
            discrete=is_discrete,
            env_evaluation_trials=config["env_evaluation_trials"]
        )


        for algo_item in config["algorithms"]:
            for data_ratio in config["data_ratio"]:
                try:
                    algo = utils.get_algo(algo_item["name"], is_discrete)
                except ValueError:
                    pass

                try:
                    train_results = train(full_dataset = full_dataset,
                                          data_ratio = data_ratio,
                                          env = env,
                                          env_limits = env_limits,
                                          is_discrete = is_discrete,
                                          ds_size = ds_size,
                                          algo = algo,
                                          algo_args = algo_item.get("args", {}),
                                          config = config,
                                          scorers = scorers,
                                          seed = seed)

                    for epoch, metric in train_results:
                        results.append(
                            utils.merge_dicts(
                                {
                                    "env":  env.unwrapped.spec.id,
                                    "algo": "{} {}".format(data_ratio, algo.__name__.replace("Discrete", "")),
                                    "epoch": epoch,
                                },
                                metric,
                            )
                        )
                except Exception:
                    error = "[ERROR] {}: algo {} on env {}. Trace: {}".format(
                        datetime.datetime.now(), algo.__name__, env.unwrapped.spec.id, traceback.format_exc())
                    print(error, file=sys.stderr)
                    thrown_errors.append(error)
                    
        #for each environment, save a different file
        save_results(results, config.get("scorers", []), "results/{}.parquet".format(env.unwrapped.spec.id))

    return thrown_errors




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--config",
        help="Path to the experiment yaml config file",
        type=str,
        default="config.yaml",
    )

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    errors = run(config)
    for e in errors:
        print(e)
    if len(errors) == 0:
        print("ALL DONE!")