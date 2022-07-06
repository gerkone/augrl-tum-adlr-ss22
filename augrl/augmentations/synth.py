"""
This module contains some augmentation methods, to be used in training
"""

from typing import List, Tuple, Dict

import numpy as np
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.dataset import Transition
from d3rlpy.iterators import RandomIterator
from d3rlpy.models.torch.q_functions.ensemble_q_function import (
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
)
from d3rlpy.torch_utility import TorchMiniBatch
import gym

def gaussian(
    transitions: List[Transition], scaling: np.ndarray, limits: Dict[str, np.array], sigma: float = 1e-3
) -> List[Transition]:
    def _gaussian(trans: Transition, scaling: np.ndarray, sigma: float):
        augmented_obs = trans.observation + scaling * np.random.normal(
            loc=0, scale=sigma, size=trans.get_observation_shape()
        )
        augmented_obs = clip_observation(augmented_obs, limits)
        return Transition(
            trans.get_observation_shape(),
            trans.get_action_size(),
            augmented_obs,
            trans.action,
            trans.reward,
            trans.next_observation,
            trans.terminal,
        )

    return [_gaussian(trans, scaling, sigma) for trans in transitions]


def uniform(
    transitions: List[Transition], scaling: np.ndarray, limits: Dict[str, np.array], alpha: float = 1e-3
) -> List[Transition]:
    def _uniform(trans: Transition, scaling: np.ndarray, alpha: float) -> Transition:
        augmented_obs = trans.observation + scaling * np.random.uniform(
            loc=-alpha, scale=alpha, size=trans.get_observation_shape()
        )
        augmented_obs = clip_observation(augmented_obs, limits)
        return Transition(
            trans.get_observation_shape(),
            trans.get_action_size(),
            augmented_obs,
            trans.action,
            trans.reward,
            trans.next_observation,
            trans.terminal,
        )

    return [_uniform(trans, scaling, alpha) for trans in transitions]


def mixup(
    transitions: List[Transition], scaling: np.ndarray, limits: Dict[str, np.array], eps: float = 0.4
) -> List[Transition]:
    def _mixup(trans: Transition, eps: float) -> Transition:
        gamma = np.random.beta(eps, eps, size=trans.get_observation_shape())
        # s_t = gamma * s_t + (1 - gamma) * s_{t+1}
        augmented_obs = gamma * trans.observation * (1 - gamma) * trans.next_observation
        augmented_obs = clip_observation(augmented_obs, limits)
        return Transition(
            trans.get_observation_shape(),
            trans.get_action_size(),
            augmented_obs,
            trans.action,
            trans.reward,
            trans.next_observation,
            trans.terminal,
        )

    # scaling not needed for mixup
    _ = scaling
    return [_mixup(trans, eps) for trans in transitions]


def adversarial(
    transitions: List[Transition],
    scaling: np.ndarray,
    limits: Dict[str, np.array],
    norm: str = "2",
    eps: float = 1e-4,
    impl: TorchImplBase = None,
    batch_size: int = 2048,
) -> List[Transition]:
    def _adversarial(
        batch: TorchMiniBatch,
        impl: TorchImplBase,
        norm: str,
        eps: float,
        batch_size: int,
        state_size: Tuple,
        action_size: Tuple,
    ) -> List[Transition]:
        impl.q_function.requires_grad_(False)
        batch.observations.requires_grad = True
        # single-step projected gradient attack (PGD)
        if isinstance(impl.q_function, EnsembleContinuousQFunction):
            action = impl._predict_best_action(batch.observations)
            q_val = impl.q_function(
                batch.observations, action 
            )
        if isinstance(impl.q_function, EnsembleDiscreteQFunction):
            q_val = impl.q_function(batch.observations).max(axis=-1).values

        # each batch dimension only accounts for one direction of the resulting gradient
        q_val = q_val.sum()

        q_val.backward()

        obs_grad = batch.observations.grad
        batch.observations.requires_grad = False
        impl.q_function.requires_grad_(True)

        if norm == "inf":
            # FSGM
            augmented_observations = batch.observations + eps * obs_grad.sign()

        else:
            augmented_observations = (
                batch.observations.grad
                + eps
                * obs_grad
                / obs_grad.flatten(1, -1).norm(p=int(norm), dim=-1).view(-1, 1, 1, 1)
            )
        np_augmented_observations = augmented_observations.detach().cpu().numpy()
        np_actions = batch.actions.detach().cpu().numpy()
        np_rewards = batch.rewards.detach().cpu().numpy()
        np_next_observations = batch.next_observations.detach().cpu().numpy()
        np_terminals = batch.terminals.detach().cpu().numpy()
        del batch
        del augmented_observations
        # fill transition list iterating over elements of the batch
        # TODO slow!
        np_augmented_observations = [clip_observation(augmented_obs, limits) for augmented_obs in np_augmented_observations]
        return [
            Transition(
                state_size,
                action_size,
                np_augmented_observations[i],
                np_actions[i],
                np_rewards[i],
                np_next_observations[i],
                np_terminals[i],
            )
            for i in range(batch_size)
        ]

    # scaling not needed for adversarial
    _ = scaling
    if (
        impl is None
        or not isinstance(impl, TorchImplBase)
        or not (hasattr(impl, "q_function"))
    ):
        raise ValueError(
            "Trying to perform adversarial augmentation without a valid implementation"
        )

    # in adversarial we must use batches, otherwise it's too slow
    adv_iterator = RandomIterator(
        transitions,
        n_steps_per_epoch=len(transitions) // batch_size,
        batch_size=batch_size,
    )

    state_size = transitions[0].get_observation_shape()
    action_size = transitions[0].get_action_size()

    augmented_transitions: List[Transition] = []
    for batch in adv_iterator:
        fake_batch = TorchMiniBatch(
            batch,
            device=impl.device,
            scaler=impl.scaler,
            action_scaler=impl.action_scaler,
            reward_scaler=impl.reward_scaler,
        )
        augmented_transitions.extend(
            _adversarial(
                fake_batch,
                impl,
                norm,
                eps,
                batch_size,
                state_size=state_size,
                action_size=action_size,
            )
        )
    return augmented_transitions

def clip_observation(observation: np.ndarray, limits: Dict[str, np.array]):
    return np.clip(observation, limits["obs_min"], limits["obs_max"])

def clip_action(action: np.ndarray, limits: Dict[str, np.array]):
    return np.clip(action, limits["action_min"], limits["action_max"])