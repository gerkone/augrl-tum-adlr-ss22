import argparse
import datetime
import os
import sys
import traceback
from typing import Dict, List

import d3rlpy
import pandas as pd
import yaml

from augrl import utils

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"


def run(config: Dict) -> List:
    experiment_date = datetime.datetime.now().strftime("%d%m%Y")
    thrown_errors = []
    results = []
    d3rlpy.seed(config["seed"])

    for env_item in config["environments"]:
        try:
            full_dataset, env = d3rlpy.datasets.get_dataset(env_item["name"])
        except ValueError:
            try:
                full_dataset, env = d3rlpy.datasets.get_d4rl(env_item["name"])
            except ValueError:
                continue
        experiment_algos = (
            config["algorithms_discrete"]
            if env_item["discrete"]
            else config["algorithms_continuous"]
        )
        for algo_item in experiment_algos:
            for data_ratio in algo_item["data_ratio"]:
                try:
                    env.reset(seed=config["seed"])
                except TypeError:
                    print("Could not set seed for {}".format(env_item["name"]))
                algo = utils.get_algo(algo_item["name"], env_item["discrete"])
                try:
                    dataset = utils.trim(full_dataset, data_ratio)
                    agent = algo(
                        use_gpu=config["cuda"],
                        augmentations=config["augmentations"],
                        real_ratio=config["real_ratio"],
                        scaler=config["scaler"],
                        **algo_item.get("args", {}),
                    )
                    agent.generated_maxlen = len(dataset.observations)
                    metrics = agent.fit(
                        dataset=dataset,
                        eval_episodes=full_dataset,
                        n_steps=env_item["batches"],
                        n_steps_per_epoch=config["steps_per_epoch"],
                        scorers=(
                            utils.merge_dicts(
                                {
                                    name: getattr(d3rlpy.metrics, name)
                                    for name in algo_item.get("scorers", [])
                                },
                                {
                                    "environment_reward": d3rlpy.metrics.evaluate_on_environment(
                                        env, n_trials=config["env_evaluation_trials"]
                                    )
                                },
                            )
                        ),
                        verbose=config["verbose"],
                        show_progress=config["show_progress"],
                        save_interval=config["save_interval"],
                        with_timestamp=False,
                        logdir="d3rlpy_logs/{}_{}".format(
                            algo.__name__, experiment_date
                        ),
                        experiment_name="{}_{}_{}_{}".format(
                            data_ratio,
                            env_item["name"],
                            algo.__name__,
                            env_item["batches"],
                        ),
                    )
                    # easy pandasify
                    for epoch, metric in metrics:
                        results.append(
                            utils.merge_dicts(
                                {
                                    "env": env_item["name"],
                                    "algo": "{} {}".format(data_ratio, algo.__name__),
                                    "epoch": epoch,
                                },
                                metric,
                            )
                        )
                    results_df = pd.DataFrame(results)
                    results_df.to_parquet(
                        "results/cloud_results_{}.parquet".format(
                            datetime.datetime.now().strftime("%d%m%Y_%H%M")
                        )
                    )
                except Exception:
                    error = "[ERROR] {}: algo {} on env {}. Trace: {}".format(
                        datetime.datetime.now(),
                        algo.__name__,
                        env_item["name"],
                        traceback.format_exc(),
                    )
                    print(error, file=sys.stderr)
                    thrown_errors.append(error)
    return thrown_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--config",
        help="Path to the experiment yaml config file",
        type=str,
        default="experiments/configs/config.yaml",
    )

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    errors = run(config)
    for e in errors:
        print(e)
    if len(errors) == 0:
        print("ALL DONE!")
