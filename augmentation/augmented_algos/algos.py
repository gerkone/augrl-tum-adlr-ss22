from typing import Dict, List, Tuple

import d3rlpy

from .base import custom_augmented_fitter


class AugmentedBCQ(d3rlpy.algos.BCQ):
    def __init__(self, *args, augmentations: List[Tuple[str, Dict]] = None, **kwargs):
        self.augmentations = augmentations if augmentations is not None else []
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedBCQ)
        super().__init__(*args, **kwargs)


class AugmentedDiscreteBCQ(d3rlpy.algos.DiscreteBCQ):
    def __init__(self, *args, augmentations: List[Tuple[str, Dict]] = None, **kwargs):
        self.augmentations = augmentations if augmentations is not None else []
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedDiscreteBCQ)
        super().__init__(*args, **kwargs)
