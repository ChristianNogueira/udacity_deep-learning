import numpy as np


# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    return -1 * sum([y * np.log(p) + (1 - y) * np.log(1 - p) for y, p in zip(Y, P)])
