import numpy as np

def sumSq(outputs, targets, d=False):
    if d:
        return outputs - targets
    return np.sum(np.power(outputs-targets, 2))/2
