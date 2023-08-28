import numpy as np
from sklearn.linear_model import LinearRegression

def linear(*args, n, batch, n0):
    a = args[0]
    b = args[1]
    
    nt = n - n0
    x = np.arange(n0, n)/n
    time = a + b*x

    time = np.concatenate((np.repeat(0, n0), time))

    if batch > 1:
        time = get_batched_time(n, batch, time)
        
    return time


def quadratic(*args, n, batch, n0):
    a = args[0]
    b = args[1]
    c = args[2]
    
    nt = n - n0
    x = np.arange(n0, n)/n
    time = a + b*x + c*x**2

    time = np.concatenate((np.repeat(0, n0), time))

    if batch > 1:
        time = get_batched_time(n, batch, time)
        
    return time

def constant(*args, n, batch, n0):
    a = args[0]
    nt = n - n0
    time = a * np.repeat(1, nt)
    time = np.concatenate((np.repeat(0, n0), time))
    
    if batch > 1:
        time = get_batched_time(n, batch, time)
        
    return time

def regress(*args, n, batch, n0):
    x = args[0][:, None]
    y = args[1]
    xtest = args[2][:, None]
    
    X     = np.concatenate((x, x**2), axis=1)
    Xtest = np.concatenate((xtest, xtest**2), axis=1)
    reg   = LinearRegression().fit(X, y)
    ytest = reg.predict(Xtest)
    
    if batch > 1:
        ytest = get_batched_time(n, batch, ytest)
        
    return ytest

def get_batched_time(n, b, time):
    timenew = np.repeat(0.001, n)
    timenew[0::b] = time[0::b]
    #time = time[0::b]
    #time = np.repeat(time, b)
    return timenew