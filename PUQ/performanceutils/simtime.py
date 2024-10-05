import numpy as np


def linear(*args, n, seed):
    a = args[0]
    b = args[1]
    x = np.arange(0, n) / n
    time = a + b * x
    return time


def normal(*args, n, seed):
    np.random.seed(seed)
    a = args[0]
    b = args[1]
    cons = args[2]

    time = np.random.normal(loc=a, scale=b, size=n)
    time[time < 0] = cons
    return time


def constant(*args, n, seed):
    a = args[0]

    time = a * np.repeat(1, n)

    return time
