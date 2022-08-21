# AugRL: offline RL, with little data

Team project repo for the TUM course _Advanced Deep Learning for Robotics_ (ADLR).

Our project focuses on investigating the impact of data on generalization and performance of offline RL algorithms.

Reinforcement learning sometimes suffers from costly data collection (labeling datasets with rewards usually requires human supervision). We want to address this problem by implementing augmentation techniques on the already collected datasets.

## Installation

```
pip install -r requirements.txt
```

## Milestones
The key steps for the project are the following:
- Implement data augmentation techniques
- Test and benchmark the selected offline algorithms in the Mujoco environments (half-cheetah, humanoid), with standard and augmented datasets
- Exploratory data and downstream reward labeling with a learned reward function

## Data augmentation
- __Additive noise__: eg uniform, Gaussian
- __State mixup__
- __Adversarial state training__

## Related work
- [d3rlpy](https://github.com/takuseno/d3rlpy) provides high quality implementations of state of the art offline and online RL algorithms. We picked it as our codebase
- [D4RL](https://github.com/rail-berkeley/d4rl) provided a great approach on benchmarking and comparing offline RL algorithms, with special regard towards data
- [S4RL](https://arxiv.org/abs/2103.06326) explored data augmentation as self supervision, although not with the purpose of specifically evaluating generalization
- [ExORL](https://arxiv.org/abs/2201.13425) presented a way to manufacture rollouts artificially and assign a reward downstream
