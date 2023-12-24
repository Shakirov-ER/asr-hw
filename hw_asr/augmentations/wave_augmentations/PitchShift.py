from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply
import torch_audiomentations as ta


class PitchShift(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug = ta.augmentations.PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self.aug.apply_transform(data)


class RandomPitchShift(AugmentationBase):
    def __init__(self, p=0.5, *args, **kwargs):
        self._aug = RandomApply(PitchShift(*args, **kwargs), p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)
