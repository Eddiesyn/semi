import numpy as np


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):

    def __call__(self, target):
        return target['label']


class VideoID(object):

    def __call__(self, target):
        return target['video_id']


class ClassLabel_fromarray(object):
    def __init__(self, label_array):
        """Initialize target transform

        :param label_array: the array which contains labels of all samples
        """
        self.labels = label_array

    def __call__(self, index):
        return self.labels[index]
