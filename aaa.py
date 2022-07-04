import d3rlpy
from augrl.utils import m_evaluate_on_environment
from d3rlpy.metrics import evaluate_on_environment
from augrl.algos import AugmentedBCQ
from torch.multiprocessing import set_start_method
import os
import gym

gym.logger.set_level(gym.logger.ERROR)
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"


if __name__ == "__main__":
    set_start_method("spawn")
    dataset, env = d3rlpy.datasets.get_d4rl("walker2d-random-v0")
    algo = AugmentedBCQ(use_gpu=False)
    limits = {}
    limits["obs_min"] = env.observation_space.low
    limits["obs_max"] = env.observation_space.high
    algo.limits = limits
    metrics = algo.fit(
        dataset=dataset,
        eval_episodes=dataset,
        n_steps=10,
        n_steps_per_epoch=5,
        scorers=
            {
                "environment_reward": evaluate_on_environment(
                    env = env,
                    n_trials=21,
                    
                )
            },
    )