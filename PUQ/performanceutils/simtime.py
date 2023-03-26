import numpy as np

def linear(*args, n):
    a = args[0]
    b = args[1]
    x = np.arange(0, n)/n
    time = a + b*x
    return time

def normal(*args, n):
    a = args[0]
    b = args[1]
    time = np.random.normal(loc=a, scale=b, size=n)
    time[time < 0] = 0.00001
    return time
