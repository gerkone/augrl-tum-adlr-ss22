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
    AugmentedBEAR,
    AugmentedCQL,
    AugmentedDiscreteBC,
    AugmentedDiscreteBCQ,
    AugmentedDiscreteCQL,
)

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

USE_GPU = False
REAL_RATIO = 0.5
AUGMENTATIONS = [("gaussian", {"sigma": 1e-3}), ("mixup", {"eps": 0.4})]
ENVS = [
    {"name": "cartpole-replay", "discrete": True, "epochs": 60},
    {"name": "hammer-expert-v0", "discrete": False, "epochs": 50},
    {"name": "halfcheetah-medium-replay-v0", "discrete": False, "epochs": 50},
    {"name": "hopper-medium-replay-v0", "discrete": False, "epochs": 40},
]
ALGOS_C = [
    {
        "algo_class": AugmentedBC,
        "config": {"batch_size": 300},
        "scorers": {},
    },
    {
        "algo_class": AugmentedBEAR,
        "config": {"batch_size": 500},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": AugmentedBCQ,
        "config": {"batch_size": 300},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": AugmentedCQL,
        "config": {"batch_size": 100},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.BC,
        "config": {"batch_size": 300},
        "scorers": {},
    },
    {
        "algo_class": d3rlpy.algos.BEAR,
        "config": {"batch_size": 500},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.BCQ,
        "config": {"batch_size": 300},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.CQL,
        "config": {"batch_size": 100},
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
        "config": {"batch_size": 300},
        "scorers": {},
    },
    {
        "algo_class": AugmentedDiscreteBCQ,
        "config": {"batch_size": 300},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": AugmentedDiscreteCQL,
        "config": {"batch_size": 100},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.BC,
        "config": {"batch_size": 300},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.DiscreteBCQ,
        "config": {"batch_size": 300},
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.DiscreteCQL,
        "config": {"batch_size": 100},
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
                    | {"environment_reward": evaluate_on_environment(env, n_trials=3)}
                ),
                verbose=False,
                experiment_name="{}_{}_{}_clean".format(
                    env_item["name"], algo.__name__, env_item["epochs"]
                ),
            )
            # easy pandasify
            for epoch, metric in metrics:
                results.append(
                    {
                        "env": env_item["name"],
                        "algo": "{}_clean".format(algo.__name__),
                        "epoch": epoch,
                    }
                    | metric
                )
    results_df = pd.DataFrame(results)
    results_df.to_parquet(
        "results/results_{}.parquet".format(
            datetime.datetime.now().strftime("%d%m%Y_%H%M")
        )
    )
