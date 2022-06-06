"""Modified fitter method, ready for monkey patching the original algos"""

import random
from collections import defaultdict
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast

import numpy as np
from d3rlpy.base import LearnableBase
from d3rlpy.constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    ActionSpace,
)
from d3rlpy.dataset import Episode, MDPDataset, Transition
from d3rlpy.iterators import RandomIterator, RoundIterator, TransitionIterator
from d3rlpy.logger import LOG
from tqdm import tqdm

import augrl.augmentations.synth


def custom_augmented_fitter(
    cls,
    dataset: Union[List[Episode], List[Transition], MDPDataset],
    n_epochs: Optional[int] = None,
    n_steps: Optional[int] = None,
    n_steps_per_epoch: int = 10000,
    save_metrics: bool = True,
    experiment_name: Optional[str] = None,
    with_timestamp: bool = True,
    logdir: str = "d3rlpy_logs",
    verbose: bool = True,
    show_progress: bool = True,
    tensorboard_dir: Optional[str] = None,
    eval_episodes: Optional[List[Episode]] = None,
    save_interval: int = 1,
    scorers: Optional[Dict[str, Callable[[Any, List[Episode]], float]]] = None,
    shuffle: bool = True,
    callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """Iterate over epochs steps to train with the given dataset, with augmentation"""
    # check augmentations
    if not hasattr(cls, "augmentations"):
        if verbose:
            LOG.debug("'augmentations' class field ot set. Defaulting to only clean")
        cls.augmentations = []
    transitions = []
    if isinstance(dataset, MDPDataset):
        for episode in dataset.episodes:
            transitions += episode.transitions
    elif not dataset:
        raise ValueError("empty dataset is not supported.")
    elif isinstance(dataset[0], Episode):
        for episode in cast(List[Episode], dataset):
            transitions += episode.transitions
    elif isinstance(dataset[0], Transition):
        transitions = list(cast(List[Transition], dataset))
    else:
        raise ValueError(f"invalid dataset type: {type(dataset)}")

    # check action space
    if cls.get_action_type() == ActionSpace.BOTH:
        pass
    elif transitions[0].is_discrete:
        assert (
            cls.get_action_type() == ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        assert (
            cls.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

    iterator: TransitionIterator
    if n_epochs is None and n_steps is not None:
        assert n_steps >= n_steps_per_epoch
        n_epochs = n_steps // n_steps_per_epoch
        iterator = RandomIterator(
            transitions,
            n_steps_per_epoch,
            batch_size=cls._batch_size,
            n_steps=cls._n_steps,
            gamma=cls._gamma,
            n_frames=cls._n_frames,
            real_ratio=cls._real_ratio,
            generated_maxlen=cls._generated_maxlen,
        )
        if verbose:
            LOG.debug("RandomIterator is selected.")
    elif n_epochs is not None and n_steps is None:
        iterator = RoundIterator(
            transitions,
            batch_size=cls._batch_size,
            n_steps=cls._n_steps,
            gamma=cls._gamma,
            n_frames=cls._n_frames,
            real_ratio=cls._real_ratio,
            generated_maxlen=cls._generated_maxlen,
            shuffle=shuffle,
        )
        if verbose:
            LOG.debug("RoundIterator is selected.")
    else:
        raise ValueError("Either of n_epochs or n_steps must be given.")

    # setup logger
    logger = cls._prepare_logger(
        save_metrics,
        experiment_name,
        with_timestamp,
        logdir,
        verbose,
        tensorboard_dir,
    )

    # add reference to active logger to algo class during fit
    cls._active_logger = logger

    # initialize scaler
    if cls._scaler:
        if verbose:
            LOG.debug("Fitting scaler...", scaler=cls._scaler.get_type())
        cls._scaler.fit(transitions)

    # initialize action scaler
    if cls._action_scaler:
        if verbose:
            LOG.debug(
                "Fitting action scaler...",
                action_scaler=cls._action_scaler.get_type(),
            )
        cls._action_scaler.fit(transitions)

    # initialize reward scaler
    if cls._reward_scaler:
        if verbose:
            LOG.debug(
                "Fitting reward scaler...",
                reward_scaler=cls._reward_scaler.get_type(),
            )
        cls._reward_scaler.fit(transitions)

    # instantiate implementation
    if cls._impl is None:
        if verbose:
            LOG.debug("Building models...")
        transition = iterator.transitions[0]
        action_size = transition.get_action_size()
        observation_shape = tuple(transition.get_observation_shape())
        cls.create_impl(cls._process_observation_shape(observation_shape), action_size)
        if verbose:
            LOG.debug("Models have been built.")
    else:
        LOG.warning("Skip building models since they're already built.")

    # save hyperparameters
    cls.save_params(logger)
    # refresh evaluation metrics
    cls._eval_results = defaultdict(list)
    # refresh loss history
    cls._loss_history = defaultdict(list)
    # selected augmentation functions
    augmentation_functions = [
        (getattr(augrl.augmentations.synth, fn), args) for fn, args in cls.augmentations
    ]

    transitions_ = iterator.transitions.copy()

    # training loop
    total_step = 0
    for epoch in range(1, n_epochs + 1):

        # dict to add incremental mean losses to epoch
        epoch_loss = defaultdict(list)

        range_gen = tqdm(
            range(len(iterator)),
            disable=not show_progress,
            desc=f"Epoch {int(epoch)}/{n_epochs}",
        )

        iterator.reset()

        new_transitions = _augment(
            augmentation_functions, transitions_, cls._generated_maxlen
        )
        if new_transitions:
            iterator.add_generated_transitions(new_transitions)
            if verbose:
                LOG.debug(
                    f"{len(new_transitions)} transitions are augmented.",
                    real_transitions=len(iterator.transitions),
                    fake_transitions=len(iterator.generated_transitions),
                )
        for itr in range_gen:
            with logger.measure_time("step"):
                # pick transitions
                with logger.measure_time("sample_batch"):
                    batch = next(iterator)

                with logger.measure_time("algorithm_update"):
                    # update parameters
                    loss = cls.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {k: np.mean(v) for k, v in epoch_loss.items()}
                        range_gen.set_postfix(mean_loss)

            total_step += 1

            # call callback if given
            if callback:
                callback(cls, epoch, total_step)

        # save loss to loss history dict
        cls._loss_history["epoch"].append(epoch)
        cls._loss_history["step"].append(total_step)
        for name, vals in epoch_loss.items():
            if vals:
                cls._loss_history[name].append(np.mean(vals))

        if scorers and eval_episodes:
            cls._evaluate(eval_episodes, scorers, logger)

        # save metrics
        metrics = logger.commit(epoch, total_step)

        # save model parameters
        if epoch % save_interval == 0:
            logger.save_model(total_step, cls)

        yield epoch, metrics

    # drop reference to active logger since out of fit there is no active
    # logger
    cls._active_logger.close()
    cls._active_logger = None


def _augment(
    augmentation_functions: List[Tuple[Callable, Dict]],
    transitions: List[Transition],
    maxlen: int,
):
    """Generate new transitions with augumentations"""
    new_transitions = []
    n_augmentable = np.min([len(transitions), maxlen]) // len(augmentation_functions)
    random.shuffle(transitions)
    augmentable_portions = [
        transitions[i * n_augmentable : (i + 1) * n_augmentable]
        for i in range(len(augmentation_functions))
    ]
    for (fn, args), transitions_i in zip(augmentation_functions, augmentable_portions):
        new_transitions.extend(
            augrl.augmentations.synth.generate_new_data(transitions_i, fn, args)
        )
    return new_transitions
