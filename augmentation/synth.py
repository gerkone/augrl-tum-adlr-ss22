"""
This module contains some augmentation methods, to be used in training
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from d3rlpy.dataset import Transition, TransitionMiniBatch


def augmenter_wrapper(
    fn: Callable, batch: TransitionMiniBatch, args: Dict
) -> TransitionMiniBatch:
    augmented_transervations = [fn(trans, **args) for trans in batch.transitions]
    return TransitionMiniBatch(augmented_transervations)


def generate_new_data(
        transitions: List[Transition], fn: Callable, args: Dict
    ) -> Optional[List[Transition]]:
    return [fn(trans, **args) for trans in transitions]

def clean(transition: Transition) -> Transition:
    return transition


def gaussian(trans: Transition, sigma: float = 1e-3) -> Transition:
    augmented_obs = trans.observation + np.random.normal(
        loc=0, scale=sigma, size=trans.get_observation_shape()
    )
    return Transition(
        trans.get_observation_shape(),
        trans.get_action_size(),
        augmented_obs,
        trans.action,
        trans.reward,
        trans.next_observation,
        trans.terminal,
    )


def uniform(trans: Transition, alpha: float = 1e-3) -> Transition:
    augmented_obs = trans.observation + np.random.uniform(
        loc=-alpha, scale=alpha, size=trans.get_observation_shape()
    )
    return Transition(
        trans.get_observation_shape(),
        trans.get_action_size(),
        augmented_obs,
        trans.action,
        trans.reward,
        trans.next_observation,
        trans.terminal,
    )


def mixup(trans: Transition, eps: float = 0.4) -> Transition:
    gamma = np.random.beta(eps, eps, size=trans.get_observation_shape())
    # s_t = gamma * s_t + (1 - gamma) * s_{t+1}
    augmented_obs = gamma * trans.observation * (1 - gamma) * trans.next_observation
    return Transition(
        trans.get_observation_shape(),
        trans.get_action_size(),
        augmented_obs,
        trans.action,
        trans.reward,
        trans.next_observation,
        trans.terminal,
    )
