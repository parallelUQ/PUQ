import numpy as np
from sklearn.linear_model import LinearRegression

def linear(*args, n, batch):
    a = args[0]
    b = args[0]
    x = np.arange(0, n)/n
    time = a + b*x
    
    if batch > 1:
        time = get_batched_time(n, batch, time)
        
    return time

def constant(*args, n, batch):
    a = args[0]
    time = a * np.repeat(1, n)
    
    if batch > 1:
        time = get_batched_time(n, batch, time)
        
    return time

def regress(*args, n, batch):
    x = args[0][:, None]
    y = args[1]
    xtest = args[2][:, None]
    
    X     = np.concatenate((x, x**2), axis=1)
    Xtest = np.concatenate((xtest, xtest**2), axis=1)
    reg   = LinearRegression().fit(X, y)
    ytest = reg.predict(Xtest)
    return ytest

def get_batched_time(n, b, time):
    timenew = np.repeat(0.01, n)
    timenew[0::b] = time[0::b]
    #time = time[0::b]
    #time = np.repeat(time, b)
    return timenew