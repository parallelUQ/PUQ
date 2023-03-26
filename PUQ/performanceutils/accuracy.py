import numpy as np
import scipy

def exponential(*args, n, batch):
    a = args[0]
    b = args[1]
    x = np.arange(0, n)/n
    accu = a*(x**b)
    accu = [litem + 1 for litem in accu]
    

    if batch > 1:
        accu = get_batched_accuracy(n, batch, accu)
    return accu

def regress(*args, n, batch):
    x = args[0]
    y = args[1]
    xtest = args[2]
    
    xinit = [0, 0, 0, 0, 0]
    res     = scipy.optimize.minimize(optfunc, method='BFGS', x0=xinit, args={'xdata': x, 'ydata': y})    
    accu    = func(res.x, xtest)
    
    
    if batch > 1:
        accu = get_batched_accuracy(n, batch, accu)
    
    return accu
    
def func(x, xdata):
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    e = x[4]
    fu = a * xdata + b * (xdata ** 2) + c * np.exp(d * xdata) + e
    return fu

def optfunc(x, args):
    xdata = args['xdata']
    ydata = args['ydata']
    f = func(x, xdata)
    return np.sum((f - ydata)**2)

def get_batched_accuracy(n, b, acqlist):
    accu     = [acqlist[idacc] for idacc in np.arange(b-1, n, b)]
    accu_all = np.repeat(accu[0:-1], b)
    accu_all = np.concatenate((np.repeat(1, b-1), accu_all))
    accu_all = np.concatenate((accu_all, np.array([accu[-1]])))
    return accu_all