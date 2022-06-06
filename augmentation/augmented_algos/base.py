"""Modified fitter method, ready for monkey patching the original algos"""

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

import augmentation.synth


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
        LOG.debug("'augmentations' class field ot set. Defaulting to only clean")
        cls.augmentations = []
    #cls.augmentations.append(("clean", {}))
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
        LOG.debug("Fitting scaler...", scaler=cls._scaler.get_type())
        cls._scaler.fit(transitions)

    # initialize action scaler
    if cls._action_scaler:
        LOG.debug(
            "Fitting action scaler...",
            action_scaler=cls._action_scaler.get_type(),
        )
        cls._action_scaler.fit(transitions)

    # initialize reward scaler
    if cls._reward_scaler:
        LOG.debug(
            "Fitting reward scaler...",
            reward_scaler=cls._reward_scaler.get_type(),
        )
        cls._reward_scaler.fit(transitions)

    # instantiate implementation
    if cls._impl is None:
        LOG.debug("Building models...")
        transition = iterator.transitions[0]
        action_size = transition.get_action_size()
        observation_shape = tuple(transition.get_observation_shape())
        cls.create_impl(cls._process_observation_shape(observation_shape), action_size)
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
        (getattr(augmentation.synth, fn), args) for fn, args in cls.augmentations
    ]

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

        #######################################################################
        #Generate new transitions with augumentations
        new_transitions = []
        for fn, args in augmentation_functions:
            new_transitions.append(
                augmentation.synth.generate_new_data(iterator.transitions, fn, args)
            )
            
        if new_transitions:
            flat_list_transitions = [x for xs in new_transitions for x in xs]  #flatten list
            iterator.add_generated_transitions(flat_list_transitions)
            LOG.debug(
                f"{len(flat_list_transitions)} transitions are augmented.",
                real_transitions=len(iterator.transitions),
                fake_transitions=len(iterator.generated_transitions),
            )
        #######################################################################
        for itr in range_gen:
            
            # generate new transitions with dynamics models
            new_transitions = cls.generate_new_data(
                transitions=iterator.transitions,
            )
            if new_transitions:
                iterator.add_generated_transitions(new_transitions)
                LOG.debug(
                    f"{len(new_transitions)} transitions are generated.",
                    real_transitions=len(iterator.transitions),
                    fake_transitions=len(iterator.generated_transitions),
                )

        
            #generate 

            with logger.measure_time("step"):
                # pick transitions
                with logger.measure_time("sample_batch"):
                    clean_batch = next(iterator)

                with logger.measure_time("algorithm_update"):
                    batch = clean_batch
                    

                    """
                    for fn, args in augmentation_functions:
                        # augment on batch
                        batch = augmentation.synth.augmenter_wrapper(
                            fn, clean_batch, args
                        )
                    """
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