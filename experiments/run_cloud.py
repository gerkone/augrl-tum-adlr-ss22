import datetime
import os

import d3rlpy
import pandas as pd
from d3rlpy.metrics.scorer import (
    average_value_estimation_scorer,
    discounted_sum_of_advantage_scorer,
    evaluate_on_environment,
    initial_state_value_estimation_scorer,
    td_error_scorer,
)

from augrl.algos import (
    AugmentedBC,
    AugmentedBCQ,
    AugmentedCQL,
    AugmentedDiscreteBC,
    AugmentedDiscreteBCQ,
    AugmentedDiscreteCQL,
)

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

USE_GPU = True
REAL_RATIO = 0.5
AUGMENTATIONS = [("gaussian", {"sigma": 1e-3}), ("mixup", {"eps": 0.4})]
ENVS = [
    {"name": "cartpole-replay", "discrete": True, "epochs": 60},
    {"name": "pen-expert-v0", "discrete": False, "epochs": 50},
    {"name": "halfcheetah-medium-replay-v0", "discrete": False, "epochs": 50},
    {"name": "hopper-medium-replay-v0", "discrete": False, "epochs": 40},
]
ALGOS_C = [
    {
        "algo_class": AugmentedBC,
        "scorers": {},
    },
    {
        "algo_class": AugmentedBCQ,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": AugmentedCQL,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.BC,
        "scorers": {},
    },
    {
        "algo_class": d3rlpy.algos.BCQ,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.CQL,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
]
ALGOS_D = [
    {
        "algo_class": AugmentedDiscreteBC,
        "scorers": {},
    },
    {
        "algo_class": AugmentedDiscreteBCQ,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": AugmentedDiscreteCQL,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.BC,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.DiscreteBCQ,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.DiscreteCQL,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
]

if __name__ == "__main__":
    results = []
    i = 0
    for env_item in ENVS:
        try:
            dataset, env = d3rlpy.datasets.get_dataset(env_item["name"])
        except ValueError:
            try:
                dataset, env = d3rlpy.datasets.get_d4rl(env_item["name"])
            except ValueError:
                continue
        try_algos = ALGOS_D if env_item["discrete"] else ALGOS_C
        for algo_item in try_algos:
            i = (i + 1) % 2
            try:
                algo = algo_item["algo_class"]
                config = algo_item.get("config", {})
                agent = algo(
                    use_gpu=USE_GPU,
                    augmentations=AUGMENTATIONS,
                    real_ratio=REAL_RATIO,
                    **config
                )
                metrics = agent.fit(
                    dataset=dataset,
                    eval_episodes=dataset,
                    n_epochs=env_item["epochs"],
                    scorers=(
                        algo_item["scorers"]
                        | {
                            "environment_reward": evaluate_on_environment(
                                env, n_trials=3
                            )
                        }
                    ),
                    verbose=True,
                    show_progress=False,
                    experiment_name="{}_{}_{}".format(
                        env_item["name"], algo.__name__, env_item["epochs"]
                    ),
                )
                # easy pandasify
                for epoch, metric in metrics:
                    results.append(
                        {
                            "env": env_item["name"],
                            "algo": "{}".format(algo.__name__),
                            "epoch": epoch,
                        }
                        | metric
                    )
                results_df = pd.DataFrame(results)
                results_df.to_parquet(
                    "results/cloud_results_{}_{}.parquet".format(
                        datetime.datetime.now().strftime("%d%m%Y_%H%M"), i
                    )
                )
            except Exception as e:
                print(
                    "[ERROR] {}: algo {} on env {}. Trace: {}".format(
                        datetime.datetime.now(), algo.__name__, env_item["name"], e
                    )
                )
