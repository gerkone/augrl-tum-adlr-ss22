import datetime
import os
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
from d3rlpy.preprocessing import MinMaxScaler

from augrl import utils
from augrl.algos import AugmentedBCQ

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

USE_GPU = False
REAL_RATIO = 0.5
STEPS_PER_EPOCH = 2000
AUGMENTATIONS = [("gaussian", {"sigma": 1e-3}), ("mixup", {"eps": 0.4})]
ENVS = [
    # {"name": "cartpole-replay", "discrete": True, "batches": 10},
    # {"name": "pen-cloned-v0", "discrete": False, "batches": 50},
    {"name": "halfcheetah-medium-replay-v2", "discrete": False, "batches": int(1e5)},
    # {"name": "hopper-medium-replay-v0", "discrete": False, "batches": 40},
]
ALGOS_C = [
    {
        "algo_class": AugmentedBCQ,
        "data_ratio": 0.5,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.BCQ,
        "data_ratio": 0.5,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    },
    {
        "algo_class": d3rlpy.algos.BCQ,
        "data_ratio": 1.0,
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
        "algo_class": d3rlpy.algos.DiscreteBCQ,
        "data_ratio": 1.0,
        "scorers": {
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
            "advantage": discounted_sum_of_advantage_scorer,
            "initial_state_value": initial_state_value_estimation_scorer,
        },
    }
]


def run():
    thrown_errors = []
    results = []
    i = 0
    for env_item in ENVS:
        try:
            full_dataset, env = d3rlpy.datasets.get_dataset(env_item["name"])
        except ValueError:
            try:
                full_dataset, env = d3rlpy.datasets.get_d4rl(env_item["name"])
            except ValueError:
                continue
        # need to normalize manually first, then also have a scaler
        # otherwise additive noise would get scaled too
        full_dataset, min_obs, max_obs = utils.normalize(full_dataset)
        scaler = MinMaxScaler(minimum=min_obs, maximum=max_obs)
        try_algos = ALGOS_D if env_item["discrete"] else ALGOS_C
        for algo_item in try_algos:
            d3rlpy.seed(1337)
            env.reset(seed=1337)
            i = (i + 1) % 2
            try:
                algo = algo_item["algo_class"]
                config = algo_item.get("config", {})
                dataset = utils.trim(full_dataset, algo_item["data_ratio"])
                agent = algo(
                    use_gpu=USE_GPU,
                    augmentations=AUGMENTATIONS,
                    real_ratio=REAL_RATIO,
                    scaler=scaler,
                    **config
                )
                agent.generated_maxlen = len(dataset.observations)
                metrics = agent.fit(
                    dataset=dataset,
                    eval_episodes=full_dataset,
                    n_steps=env_item["batches"],
                    n_steps_per_epoch=STEPS_PER_EPOCH,
                    scorers=(
                        algo_item["scorers"]
                        | {
                            "environment_reward": evaluate_on_environment(
                                env, n_trials=15
                            )
                        }
                    ),
                    verbose=True,
                    show_progress=True,
                    experiment_name="{}_{}_{}".format(
                        env_item["name"], algo.__name__, env_item["batches"]
                    ),
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
                        }
                        | metric
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


if __name__ == "__main__":
    errors = run()
    if len(errors) == 0:
        print("ALL DONE!")
    else:
        for e in errors:
            print(e)
