from __future__ import print_function, division

import torch
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, targets):
        for t in self.transforms:
            images, targets = t(images, targets)
        return images, targets

class Random_horisontal_flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, targets):
        if np.random.random() < self.p:
            images = torch.flip(images, [-1])
            targets[:, 2] = 1 - targets[:, 2]
        return images, targets