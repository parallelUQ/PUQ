import numpy as np
import scipy
import matplotlib.pyplot as plt


def batched(*args, n, batch, ninit):
    acclist = args[0]

    nt = n - ninit
    accu = [acclist[idacc] for idacc in np.arange(batch - 1, nt, batch)]
    accu_all = np.repeat(accu[0:-1], batch)
    accu_all = np.concatenate((np.repeat(1, batch - 1), accu_all))
    accu_all = np.concatenate((accu_all, np.array([accu[-1]])))
    accu_all = np.concatenate((np.repeat(1, ninit), accu_all))

    return accu_all


def exponential(*args, n, batch, ninit):
    a = args[0]
    b = args[1]

    nt = n - ninit
    x = np.arange(0, nt) / nt
    accu = a * (x**b)
    accu = [litem + 1 for litem in accu]

    # if batch > 1:
    #    accu = get_batched_accuracy(nt, batch, accu)

    accu = np.concatenate((np.repeat(1, ninit), accu))

    return accu


def regress(*args, n, batch, ninit):
    x = args[0]
    y = args[1]
    xtest = args[2]

    xinit = [0, 0, 0, 0, 0]
    res = scipy.optimize.minimize(
        optfunc, method="BFGS", x0=xinit, args={"xdata": x, "ydata": y}
    )
    accu = func(res.x, xtest)

    if batch > 1:
        accu = get_batched_accuracy(n, batch, accu)

    return accu


def func(x, xdata):
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    e = x[4]
    fu = a * xdata + b * (xdata**2) + c * np.exp(d * xdata) + e
    return fu


def optfunc(x, args):
    xdata = args["xdata"]
    ydata = args["ydata"]
    f = func(x, xdata)
    return np.sum((f - ydata) ** 2)


def get_batched_accuracy(n, b, acqlist):
    accu = [acqlist[idacc] for idacc in np.arange(b - 1, n, b)]
    accu_all = np.repeat(accu[0:-1], b)
    accu_all = np.concatenate((np.repeat(1, b - 1), accu_all))
    accu_all = np.concatenate((accu_all, np.array([accu[-1]])))
    return accu_all
