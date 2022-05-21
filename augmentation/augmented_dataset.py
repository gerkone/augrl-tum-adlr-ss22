"""This module contains the class MDPDatasetAugmented
"""

from typing import Callable, List, Optional, Tuple

import d3rlpy
import numpy as np

ALLOWED_AUGMENTATIONS = ["gaussian", "uniform", "mixup"]


class MDPDatasetAugmented(d3rlpy.dataset.MDPDataset):
    """Allows to use augmentation directly on a d3rlpy.dataset.MDPDataset"""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        episode_terminals: Optional[np.ndarray] = ...,
        augmentations: List[str] = None,
        **kwargs,
    ):
        self.augmentations = augmentations if augmentations is not None else []

        if not all(a in ALLOWED_AUGMENTATIONS for a in self.augmentations):
            raise NotImplementedError(
                'The passed augmentations "{}" do not exist. Allowed augmentations: {}.'.format(
                    [a for a in self.augmentations if a not in ALLOWED_AUGMENTATIONS],
                    ALLOWED_AUGMENTATIONS,
                )
            )

        # augmentation parameters
        self._w_normal = kwargs.get("w_normal", 1e-3)
        self._w_uniform = kwargs.get("w_uniform", 1e-5)
        self._alpha = kwargs.get("alpha", 0.4)
        partial_observations = [observations]
        partial_actions = [actions]
        partial_rewards = [rewards]
        partial_terminals = [terminals]
        partial_episode_terminals = [episode_terminals]

        for augmentation_name in self.augmentations:
            augment_fn = getattr(self, f"_{augmentation_name}")
            (
                aug_observations,
                aug_actions,
                aug_rewards,
                aug_terminals,
                aug_episode_terminals,
            ) = self.augment(
                augment_fn, observations, actions, rewards, terminals, episode_terminals
            )
            partial_observations.append(aug_observations)
            partial_actions.append(aug_actions)
            partial_rewards.append(aug_rewards)
            partial_terminals.append(aug_terminals)
            partial_episode_terminals.append(aug_episode_terminals)

        observations = np.concatenate(partial_observations)
        actions = np.concatenate(partial_actions)
        rewards = np.concatenate(partial_rewards)
        terminals = np.concatenate(partial_terminals)
        episode_terminals = np.concatenate(partial_episode_terminals)
        super().__init__(
            observations,
            actions,
            rewards,
            terminals,
            episode_terminals,
        )

    def augment(
        self,
        augmenter_fn: Callable[[np.ndarray], np.ndarray],
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        episode_terminals: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        aug_actions = actions.copy()
        aug_rewards = rewards.copy()
        aug_terminals = terminals.copy()
        aug_episode_terminals = episode_terminals.copy()
        aug_observations = augmenter_fn(observations.copy())
        return (
            aug_observations,
            aug_actions,
            aug_rewards,
            aug_terminals,
            aug_episode_terminals,
        )

    @classmethod
    def from_mdpdataset(
        cls, dataset: d3rlpy.dataset.MDPDataset, augmentations: List[str], **kwargs
    ) -> "MDPDatasetAugmented":
        return cls(
            dataset.observations,
            dataset.actions,
            dataset.rewards,
            dataset.terminals,
            dataset.episode_terminals,
            augmentations,
            **kwargs,
        )

    def _gaussian(self, observations: np.ndarray) -> np.ndarray:
        return observations + self._w_normal * np.random.normal(
            loc=0, scale=1, size=observations.shape
        )

    def _uniform(self, observations: np.ndarray) -> np.ndarray:
        return observations + self._w_uniform * np.random.uniform(
            a=observations.min(), b=observations.max(), size=observations.shape
        )

    def _mixup(self, observations: np.ndarray) -> np.ndarray:
        gamma = np.random.beta(self._alpha, self._alpha, size=observations[1:].shape)
        # s_t = gamma * s_t + (1 - gamma) * s_{t+1}
        observations[:-1] = gamma * observations[:-1] * (1 - gamma) * observations[1:]
        return observations
