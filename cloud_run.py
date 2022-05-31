from augmentation.augmented_dataset import MDPDatasetAugmented
import os
import d3rlpy
from d3rlpy.metrics.scorer import (
    average_value_estimation_scorer,
    td_error_scorer,
    evaluate_on_environment,
    discounted_sum_of_advantage_scorer,
    initial_state_value_estimation_scorer,

)
import seaborn as sns
import pandas as pd
import itertools

import warnings

warnings.filterwarnings("ignore")


os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

i = 0

USE_GPU = True
AUGMENTATIONS = ["gaussian", "mixup"]
# epochs per environment or per algorithm?
ENVS = [
    {"name": "cartpole-replay", "discrete": True, "epochs": 50},
    {"name": "door-human-v1", "discrete": False, "epochs": 50},
    {"name": "halfcheetah-medium-replay-v0", "discrete": False, "epochs": 50},
    {"name": "hopper-medium-replay-v0", "discrete": False, "epochs": 40}
]
ALGOS_C = [
    {
        "algo_class": d3rlpy.algos.BCQ,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer
        },
    },
    {
        "algo_class": d3rlpy.algos.CQL,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer
        },
    },
]
ALGOS_D = [
    {
        "algo_class": d3rlpy.algos.DiscreteBCQ,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer
        },
    },
    {
        "algo_class": d3rlpy.algos.DiscreteCQL,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer
        },
    },
]

results = []
# TODO save best models
for env_item in ENVS:
    try:
        dataset, env = d3rlpy.datasets.get_dataset(env_item["name"])
    except ValueError:
        try:
            dataset, env = d3rlpy.datasets.get_d4rl(env_item["name"])
        except ValueError:
            continue
    augmented_dataset = MDPDatasetAugmented.from_mdpdataset(
        dataset, augmentations=AUGMENTATIONS
    )
    try_algos = ALGOS_D if env_item["discrete"] else ALGOS_C
    for algo_item in try_algos:
        algo = algo_item["algo_class"]
        agent_clean = algo(use_gpu=USE_GPU)
        agent_augmented = algo(use_gpu=USE_GPU)
        # clean training
        metrics_clean = agent_clean.fit(
            dataset,
            eval_episodes=dataset,
            n_epochs=env_item["epochs"],
            scorers=(
                algo_item["scorers"]
                | {"environment_reward": evaluate_on_environment(env, n_trials=2)}
            ),
            verbose=False,
            show_progress=False,
            experiment_name="{}_{}_{}_clean".format(
                env_item["name"], algo.__name__, env_item["epochs"]
            ),
        )
        # augmented training
        metrics_augmented = agent_augmented.fit(
            augmented_dataset,
            eval_episodes=dataset,
            n_epochs=env_item["epochs"],
            scorers=(
                algo_item["scorers"]
                | {"environment_reward": evaluate_on_environment(env, n_trials=2)}
            ),
            verbose=False,
            show_progress=False,
            experiment_name="{}_{}_{}_augmented".format(
                env_item["name"], algo.__name__, env_item["epochs"]
            ),
        )
        # easy pandasify
        for epoch, metric in metrics_clean:
            results.append(
                {
                    "env": env_item["name"],
                    "algo": "{}_clean".format(algo.__name__),
                    "epoch": epoch,
                }
                | metric
            )

        for epoch, metric in metrics_augmented:
            results.append(
                {
                    "env": env_item["name"],
                    "algo": "{}_augmented".format(algo.__name__),
                    "epoch": epoch,
                }
                | metric
            )
        i = (i + 1) % 2
        results_df = pd.DataFrame(results)
        results_df.to_parquet("results/simple_results_{}.parquet".format(i))
