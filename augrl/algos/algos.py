import copy
from typing import Dict, List, Tuple

import d3rlpy

from .base import custom_augmented_fitter

# real_ratio: amount of real data, vs augmented data to use for training


class AugmentedBEAR(d3rlpy.algos.BEAR):
    def __init__(
        self,
        *args,
        real_ratio: float = 1.0,
        augmentations: List[Tuple[str, Dict]] = None,
        **kwargs
    ):
        self.augmentations = (
            copy.deepcopy(augmentations) if augmentations is not None else []
        )
        self._real_ratio = real_ratio
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedBEAR)
        super().__init__(*args, **kwargs)

    @property
    def generated_maxlen(self) -> int:
        return self._generated_maxlen

    @generated_maxlen.setter
    def generated_maxlen(self, generated_maxlen: int):
        self._generated_maxlen = generated_maxlen


class AugmentedBCQ(d3rlpy.algos.BCQ):
    def __init__(
        self,
        *args,
        real_ratio: float = 1.0,
        augmentations: List[Tuple[str, Dict]] = None,
        **kwargs
    ):
        self.augmentations = (
            copy.deepcopy(augmentations) if augmentations is not None else []
        )
        self._real_ratio = real_ratio
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedBCQ)
        super().__init__(*args, **kwargs)

    @property
    def generated_maxlen(self) -> int:
        return self._generated_maxlen

    @generated_maxlen.setter
    def generated_maxlen(self, generated_maxlen: int):
        self._generated_maxlen = generated_maxlen


class AugmentedDiscreteBCQ(d3rlpy.algos.DiscreteBCQ):
    def __init__(
        self,
        *args,
        real_ratio: float = 1.0,
        augmentations: List[Tuple[str, Dict]] = None,
        **kwargs
    ):
        self.augmentations = (
            copy.deepcopy(augmentations) if augmentations is not None else []
        )
        self._real_ratio = real_ratio
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedDiscreteBCQ)
        super().__init__(*args, **kwargs)

    @property
    def generated_maxlen(self) -> int:
        return self._generated_maxlen

    @generated_maxlen.setter
    def generated_maxlen(self, generated_maxlen: int):
        self._generated_maxlen = generated_maxlen


class AugmentedCQL(d3rlpy.algos.CQL):
    def __init__(
        self,
        *args,
        real_ratio: float = 1.0,
        augmentations: List[Tuple[str, Dict]] = None,
        **kwargs
    ):
        self.augmentations = (
            copy.deepcopy(augmentations) if augmentations is not None else []
        )
        self._real_ratio = real_ratio
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedCQL)
        super().__init__(*args, **kwargs)

    @property
    def generated_maxlen(self) -> int:
        return self._generated_maxlen

    @generated_maxlen.setter
    def generated_maxlen(self, generated_maxlen: int):
        self._generated_maxlen = generated_maxlen


class AugmentedDiscreteCQL(d3rlpy.algos.DiscreteCQL):
    def __init__(
        self,
        *args,
        real_ratio: float = 1.0,
        augmentations: List[Tuple[str, Dict]] = None,
        **kwargs
    ):
        self.augmentations = (
            copy.deepcopy(augmentations) if augmentations is not None else []
        )
        self._real_ratio = real_ratio
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedDiscreteCQL)
        super().__init__(*args, **kwargs)

    @property
    def generated_maxlen(self) -> int:
        return self._generated_maxlen

    @generated_maxlen.setter
    def generated_maxlen(self, generated_maxlen: int):
        self._generated_maxlen = generated_maxlen


class AugmentedBC(d3rlpy.algos.BC):
    def __init__(
        self,
        *args,
        real_ratio: float = 1.0,
        augmentations: List[Tuple[str, Dict]] = None,
        **kwargs
    ):
        self.augmentations = (
            copy.deepcopy(augmentations) if augmentations is not None else []
        )
        self._real_ratio = real_ratio
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedBC)
        super().__init__(*args, **kwargs)

    @property
    def generated_maxlen(self) -> int:
        return self._generated_maxlen

    @generated_maxlen.setter
    def generated_maxlen(self, generated_maxlen: int):
        self._generated_maxlen = generated_maxlen


class AugmentedDiscreteBC(d3rlpy.algos.DiscreteBC):
    def __init__(
        self,
        *args,
        real_ratio: float = 1.0,
        augmentations: List[Tuple[str, Dict]] = None,
        **kwargs
    ):
        self.augmentations = (
            copy.deepcopy(augmentations) if augmentations is not None else []
        )
        self._real_ratio = real_ratio
        # monkey patching of original fitter method
        self.fitter = custom_augmented_fitter.__get__(self, AugmentedDiscreteBC)
        super().__init__(*args, **kwargs)

    @property
    def generated_maxlen(self) -> int:
        return self._generated_maxlen

    @generated_maxlen.setter
    def generated_maxlen(self, generated_maxlen: int):
        self._generated_maxlen = generated_maxlen
