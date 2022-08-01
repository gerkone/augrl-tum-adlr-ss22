# Augm/home/hamzahaddaoui_h/tum-adlr-ss22-3/experiments/experiment_01/d3rlpy_logs/AugmentedBCQ_door-human-v1_10/0.1_door-human-v1_AugmentedBCQ/model_2000.ptenting datasets for Offline Reinforcement Learning

Team project repo for the TUM course _Advanced Deep Learning for Robotics_ (ADLR).

Our project focuses on investigating the impact of data on generalization and performance of offline RL algorithms.

Reinforcement learning sometimes suffers from costly data collection (labeling datasets with rewards usually requires human supervision). We want to address this problem by implementing augmentation techniques on the already collected datasets.

Finally the project will move towards generating synthetic rollouts, and "best"-trajectory selection among the artificial data using heuristics or value-based approaches.

## Installation

```
pip install -r requirements.txt
```

## Milestones
The key steps for the project are the following:

- Implement data augmentation techniques on the available datasets
- Test and benchmark the selected offline algorithms in the Mujoco environments (half-cheetah, humanoid), with standard and augmented datasets
- Extend the analysis to more complex environments, with special focus on generalization and robustness towards changes
- Implement and experiment with heuristic methods for data quality evaluation and benchmark
- Consider training using purely artificial data. This requires a generator to manufacture plausible trajectories either unsupervised or from the available offline rollouts, then label those trajectories with a reward

## Data augmentation
The planned augmentations for are
- __Additive noise__: eg uniform, Gaussian
- __State mixup__
- __Adversarial state training__
- Problem-specific augmentations, dependent on the environment

## Related work
- [d3rlpy](https://github.com/takuseno/d3rlpy) provides high quality implementations of state of the art offline and online RL algorithms. We picked it as our codebase
- [D4RL](https://github.com/rail-berkeley/d4rl) provided a great approach on benchmarking and comparing offline RL algorithms, with special regard towards data
- [S4RL](https://arxiv.org/abs/2103.06326) explored data augmentation as self supervision, although not with the purpose of specifically evaluating generalization
- [ExORL](https://arxiv.org/abs/2201.13425) presented a way to manufacture rollouts artificially and assign a reward downstream
