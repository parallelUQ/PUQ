'''
Set of functions to fit a GP to each dimension of output data
'''
import numpy as np
from hetgpy import homGP
from PUQ.surrogate import emulator

def fit(fitinfo, x, theta, f, lower=None, upper=None,
        known={}, 
        noiseControl={"g_bounds": [np.sqrt(np.finfo(float).eps), 100]}, 
        init={}, 
        covtype='Gaussian', 
        maxit=100, 
        eps=np.sqrt(np.finfo(float).eps), 
        settings={"return.Ki": True, "factr": 1e7}, 
        **kwargs):
    
    numGPs = f.shape[1]
    emulist = [dict() for x in range(0, numGPs)]
    for i in range(numGPs):
        emu = emulator(
                x=x,
                theta=np.array([[i]]),
                f=f[:,i:i+1],
                method="homGP",
                args={
                    "noiseControl": noiseControl,
                    "lower": lower,
                    "upper": upper,
                    "settings": settings,
                    "init": init,
                    "known": known,
                    "covtype": covtype,
                    "maxit": maxit,
                    "eps": eps
                }
            )
        emulist[i] = emu
    fitinfo["f"] = f
    fitinfo["emulist"] = emulist
    fitinfo["numGPs"] = numGPs
    return

def predict(predinfo,fitinfo,x,theta,thetaprime,**kws):
    numGPs = fitinfo['numGPs']
    emulist = [dict() for x in range(0, numGPs)]

    # instantiate outputs
    nr, nc = (x.shape[0],numGPs)
    for key in ('mean','var'):
        predinfo[key] = np.zeros(shape=(nr,nc),dtype=float)
    predinfo['covmat'] = [np.zeros(shape=(nr,nr),dtype=float)
                          for i in range(numGPs)
                        ]
    for i in range(numGPs):
        preds = fitinfo['emulist'][i].predict(x=x,thetaprime=thetaprime)
        predinfo['mean'][:,i] = preds._info['mean']
        predinfo['var'][:,i] = preds._info['var']
        predinfo['covmat'][i] = preds._info['covmat']

    return

def update():
    return