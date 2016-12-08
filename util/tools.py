import numpy as np


def scale(x, axis=0):
    new = x - np.mean(x, axis)
    return new / np.std(new, axis)