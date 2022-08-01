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
from gym.wrappers import Monitor
from augrl import utils
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_ratio_dataset(dataset: MDPDataset, ratio: float, size: int, is_discrete: bool) -> MDPDataset:
    indices = random.sample(range(size), int(size * ratio))

    #if dataset size is very small, we may need to introduce terminals
    terminals = np.array([dataset.episode_terminals[index] for index in indices])
    if (np.sum(terminals) == 0):
        terminals = (np.random.rand(int(size * ratio)) > 0.95).astype(int)

    return MDPDataset(
            observations = np.array([dataset.observations[index] for index in indices]),
            actions = np.array([dataset.actions[index] for index in indices]),
            rewards = np.array([dataset.rewards[index] for index in indices]),
            terminals = np.array([dataset.terminals[index] for index in indices]),
            episode_terminals = terminals,
            discrete_action = is_discrete
    )


def train(full_dataset: MDPDataset, data_ratio: float, env: gym.Env, env_limits: Dict, is_discrete: bool, ds_size: int, algo: Callable, config: Dict, seed: int):
    n_steps = config.get("steps")
    n_steps_per_epoch = int(n_steps // config.get("epochs"))
    save_interval = config.get("epochs") // config.get("save_interval")

    train_set = get_ratio_dataset(full_dataset, data_ratio, ds_size, is_discrete)

    # train-test split
    _, test_set = train_test_split(full_dataset, test_size=config['test_size'])


    vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])
    rl_encoder = d3rlpy.models.encoders.VectorEncoderFactory([400, 300])
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
    conservative_weight = 10.0

    #instantiate algorithm
    agent = algo(
        use_gpu=config["cuda"],
        augmentations=config["augmentations"],
        real_ratio=config["real_ratio"],
        scaler=config["scaler"],

        actor_learning_rate=1e-3,
        critic_learning_rate=1e-4,
        temp_learning_rate=1e-4,
        imitator_learning_rate=1e-3,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        conservative_weight=conservative_weight,
        #imitator_encoder_factory=vae_encoder,
        alpha_learning_rate=0.0,
        batch_size=256,
        lam=0.75,
        action_flexibility=0.05,
        n_action_samples=10,
    )
    agent.generated_maxlen = config.get("generated_maxlen", 100000)
    agent.limits = env_limits

    base_path = "./results/{}/seed_{}/{}_{}_".format(env.unwrapped.spec.id, seed, data_ratio, algo.__name__)

    print("[INFO]: Starting training with {} with data_ratio: {} on {} samples".format(algo.__name__, data_ratio,
                                                                                       n_steps))
    #start training
    list_metrics = []
    for epoch, metrics in agent.fitter(
            dataset=train_set,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,  # to have 100 epochs
            shuffle=False,
            verbose=False,
            save_metrics=False):

        print("[INFO]: Evaluation started")

        #if save is requested, save results in parquet and set environment for video recording
        if (epoch % save_interval == 0):
            save_results(list_metrics, env.unwrapped.spec.id)
            list_metrics = []
            path = base_path+str(epoch)
            Path(path).mkdir(parents=True, exist_ok=True)
            agent.save_model(path+"/model.pt")
            print(f"Recorded video will be saved in {path}/video")
            eval_env = Monitor(env, path+"/video", force=True)
        else:
            eval_env = env

        #compute scorers
        env_reward = d3rlpy.metrics.scorer.evaluate_on_environment(eval_env)(agent)
        td_error = d3rlpy.metrics.scorer.td_error_scorer(agent, test_set)

        #TODO: Create dictionaries in a more clever way, and append to metrics list
        td_error_dict = {'seed': seed,
                         'algo': "{} {}".format(data_ratio, algo.__name__.replace("Discrete", "")),
                        'step': n_steps_per_epoch * epoch,
                        'epoch': epoch,
                        'metric': "td_error_scorer",
                        'value': td_error
                         }
        list_metrics.append(td_error_dict)

        env_reward_dict = {'seed': seed,
                           'algo': "{} {}".format(data_ratio, algo.__name__.replace("Discrete", "")),
                           'step': n_steps_per_epoch * epoch,
                           'epoch': epoch,
                           'metric': "evaluate_on_environment",
                           'value': env_reward
                           }
        list_metrics.append(env_reward_dict)

        print("[INFO]: Algorithm: {}_{}, Epoch: {}, td_error: {}, environment_reward: {}"
              .format(data_ratio, algo.__name__, epoch, td_error, env_reward))

    if len(list_metrics) != 0:  #save last results...
        save_results(list_metrics, env.unwrapped.spec.id)


def save_results(results: List, env_name: str):
    results_df = pd.DataFrame(results)
    path = "./results/{}/data_results.parquet".format(env_name)

    Path("./results/{}".format(env_name)).mkdir(parents=True, exist_ok=True)
    try:
        if (os.path.exists(path)):
            results_df = pd.concat([pd.read_parquet(path), results_df])
        results_df.to_parquet(path)
    except:
        print("[ERROR] Saving data was not successful!")


def run(config: Dict, seed: int) -> List:
    d4rl.set_dataset_path(os.path.join(os.getcwd(), "datasets"))
    thrown_errors = []
    d3rlpy.seed(seed)

    try:
        full_dataset, _ = utils.get_dataset(config['dataset'])
        env = gym.make(config['environment'])
    except NotImplementedError:
        print("Dataset not found!")
        return []

    is_discrete = True if ("cartpole" in config['dataset']) else False

    limits = {}
    limits["obs_min"] = env.observation_space.low
    limits["obs_max"] = env.observation_space.high
    if not is_discrete:
        limits["action_min"] = env.action_space.low
        limits["action_max"] = env.action_space.high

    ds_size = len(full_dataset.observations)

    env.seed(seed = seed)

    print(f"[INFO]:  Loaded {config['dataset']} on environment {env.unwrapped.spec.id}: discrete={is_discrete}, #tot_observation={ds_size}")

    for algo_item in config["algorithms"]:
        for data_ratio in config['data_ratio']:


            try:
                algo = utils.get_algo(algo_item["name"], is_discrete)
                train(full_dataset=full_dataset,
                      data_ratio=data_ratio,
                      env=env,
                      env_limits=limits,
                      is_discrete=is_discrete,
                      ds_size=ds_size,
                      algo=algo,
                      config=config,
                      seed=seed)

            except Exception:
                error = "[ERROR] {}: algo {} on env {}. Trace: {}".format(
                    datetime.datetime.now(), algo_item['name'], env.unwrapped.spec.id, traceback.format_exc())
                print(error, file=sys.stderr)
                thrown_errors.append(error)

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
        
    seeds = [10, 20, 30]
    for seed in seeds:
        print(f"INFO: using seed {seed}")
        errors = run(config, seed)