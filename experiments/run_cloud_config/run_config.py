import yaml
import datetime
import os
import sys
import traceback

import d3rlpy
import pandas as pd
from d3rlpy.metrics.scorer import (
    average_value_estimation_scorer,
    discounted_sum_of_advantage_scorer,
    evaluate_on_environment,
    initial_state_value_estimation_scorer,
    td_error_scorer,
)

import augrl
from augrl import algos
import classes


os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

config = {}



def run():
    thrown_errors = []
    results = []
    i = 0
    seed = config["seed"]
    steps_per_epoch = config["steps_per_epoch"]
    d3rlpy.seed(seed)

    for env_item in config["environments"]:
        try:
            full_dataset, env = d3rlpy.datasets.get_dataset(env_item["name"])
        except ValueError:
            continue
        #discrete_action = full_dataset.is_action_discrete()
        discrete_action = env_item["discrete"]
        try_algos = config["algorithms_D"] if discrete_action else config["algorithms_C"]
        for algo_item in try_algos:
            for real_ratio in config["real_ratios"]:
                env.reset(seed=seed)
                i = (i + 1) % 2
                #algo = d3rlpy.algos.get_algo(algo_item["algo_class"], discrete_action)  #todo: reimplement this class
                algo = classes.get_algo(algo_item["algo_class"], discrete_action)
                try:
                    dataset = augrl.utils.trim(full_dataset, algo_item["data_ratio"])
                    agent = algo(
                        use_gpu=config["use_gpu"],
                        augmentations=config["augmentations"],
                        real_ratio=real_ratio,
                        scaler="min_max"
                        #**config["algo_config"]
                    )
                    agent.generated_maxlen = len(dataset.observations)
                    metrics = agent.fit(
                        dataset=dataset,
                        eval_episodes=full_dataset,
                        n_steps=env_item["batches"],
                        n_steps_per_epoch=steps_per_epoch,
                        scorers=(
                            classes.get_scorers(algo_item["scorers"]) | 
                            {"environment_reward": evaluate_on_environment(env, n_trials=1)} 
                        ),
                        verbose=True,
                        show_progress=True,
                        experiment_name="{}_{}_{}".format(
                            env_item["name"], algo.__name__, env_item["batches"]),
                    )
                    # easy pandasify
                    for epoch, metric in metrics:
                        results.append(
                            {
                                "env": env_item["name"],
                                "algo": "{} {}".format(
                                    algo_item["data_ratio"], algo.__name__
                                ),
                                "epoch": epoch,
                            } | 
                            metric
                        )
                    results_df = pd.DataFrame(results)
                    results_df.to_parquet(
                        "results/cloud_results_{}_{}.parquet".format(
                            datetime.datetime.now().strftime("%d%m%Y_%H%M"), i
                        )
                    )
                except Exception:
                    error = "[ERROR] {}: algo {} on env {}. Trace: {}".format(
                        datetime.datetime.now(),
                        algo.__name__,
                        env_item["name"],
                        traceback.format_exc(),
                    )
                    print(error)
                    thrown_errors.append(error)
    return thrown_errors





if __name__ == '__main__':
    
    with open("experiments/run_cloud_config/config.yaml", "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    run()