# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import nibabel


class FixedNormalizer(object):
    """
    use fixed mean and stddev to normalize image intensities
    intensity = (intensity - mean) / stddev
    if clip is enabled:
        intensity = np.clip((intensity - mean) / stddev, -1, 1)
    """
    def __init__(self, mean, stddev, clip=True):
        """ constructor """
        assert stddev > 0, 'stddev must be positive'
        assert isinstance(clip, bool), 'clip must be a boolean'
        self.mean = mean
        self.stddev = stddev
        self.clip = clip

    def __call__(self, image):
        """ normalize image """
        img_np = image.get_data()
        if self.clip:
            img_np = np.clip((img_np - self.mean) / self.stddev, -1, 1)
        else:
            img_np = (img_np - self.mean) / self.stddev

        return img_np

    def to_dict(self):
        """ convert parameters to dictionary """
        obj = {'type': 0, 'mean': self.mean, 'stddev': self.stddev, 'clip': self.clip}
        return obj