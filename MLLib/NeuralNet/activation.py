import numpy as np

def sigmoid(x, d=False):
    if d:
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def tanh(x, d=False):
    if d:
        return 1-tanh(x) ** 2
    return 2/(1+np.exp(-2*x)) -1

def atan(x, d=False):
    if d:
        return 1/(x**2 + 1)
    return np.atan(x)

def relu(x, d=False):
    return (1 if d else x) if x > 0 else 0
