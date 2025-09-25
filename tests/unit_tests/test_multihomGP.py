import sys
sys.path.append('./')
import numpy as np
from scipy.stats import qmc
from PUQ.surrogate import emulator
from hetgpy import homGP
from copy import deepcopy
# create some noise 2D data
rng = np.random.default_rng(1)
lhs = qmc.LatinHypercube(d=2,rng=rng)

X = lhs.random(n=50)
# prediction grid
Xp = lhs.random(n=100)
Y = 0*X
Y[:,0] = np.sin(X[:,0])
Y[:,1] = np.cos(X[:,1])
noise = rng.normal(size=Y.shape)

Y+= noise

COVTYPE = "Matern5_2"

def test_fit():
    SETTINGS = {"return.Ki": False, "factr": 1e9}
    multiGP = emulator(x=X,theta=np.array([0,1]).reshape(2,1),f=Y,
                       method='multihomGP',args={'covtype':COVTYPE,'maxit':50,'settings':SETTINGS})
    multiGP.fit()
    GPlist = [
        homGP().mle(X=X,Z=Y[:,i],covtype=COVTYPE,maxit=50,
                    settings=SETTINGS)
        for i in range(Y.shape[1])
    ]
    emulator_keys = ['ll','theta','beta0']
    for i in range(len(GPlist)):
        GP = GPlist[i]
        for key in emulator_keys:

            assert np.allclose(multiGP._info['emulist'][i]._info[key],GP[key])
    
    return

def test_predict():
    SETTINGS = {"return.Ki": False, "factr": 1e9}
    multiGP = emulator(x=X,theta=np.array([0,1]).reshape(2,1),f=Y,
                       method='multihomGP',args={'covtype':COVTYPE,'maxit':50,'settings':SETTINGS})
    multiGP.fit()
    preds = multiGP.predict(x=Xp,thetaprime=Xp)
    GPlist = [
        homGP().mle(X=X,Z=Y[:,i],covtype=COVTYPE,
                    maxit=50,settings=SETTINGS
                ).predict(Xp,xprime=Xp)
        for i in range(Y.shape[1])
    ]
    em2GPkey = {'mean':'mean','var':'sd2'}
    for i in range(len(GPlist)):
        GP = GPlist[i]
        for em_key, GP_key in em2GPkey.items():

            assert np.allclose(preds._info[em_key][:,i],GP[GP_key])
    
    # test cov and covmat
    for i in range(len(GPlist)):
        assert np.allclose(preds._info['covmat'][i],GPlist[i]['cov'])

    return

def test_update_kriging_believer():
    SETTINGS = {"return.Ki": False, "factr": 1e9}
    multiGP = emulator(x=X,theta=np.array([0,1]).reshape(2,1),f=Y,
                       method='multihomGP',args={'covtype':COVTYPE,'maxit':50,'settings':SETTINGS})
    multiGP.fit()
    initial_lls = [multiGP._info['emulist'][i]._info['ll'] for i in range(multiGP._info['numGPs'])]
    initial_lls = deepcopy(initial_lls)
    GPlist = [
        homGP().mle(X=X,Z=Y[:,i],covtype=COVTYPE,maxit=50,
                    settings=SETTINGS)
        for i in range(Y.shape[1])
    ]
    GPlist_initial = deepcopy(GPlist)
    Xnew = X.mean(axis=0).reshape(-1,X.shape[1])
    multiGP.update(x=Xnew)
    em_keys = ['ll','theta','beta0']
    for GP in GPlist:
        Ynew = GP.predict(Xnew)['mean']
        GP.update(Xnew,Ynew,maxit=0)
    for i in range(multiGP._info['numGPs']):
        GP = GPlist[i]
        for key in em_keys:
            assert np.allclose(multiGP._info['emulist'][i]._info[key],GP[key])

def test_update():
    '''Test update and that log likelihood changes'''
    SETTINGS = {"return.Ki": False, "factr": 1e9}
    multiGP = emulator(x=X,theta=np.array([0,1]).reshape(2,1),f=Y,
                       method='multihomGP',args={'covtype':COVTYPE,'maxit':50,'settings':SETTINGS})
    multiGP.fit()
    initial_lls = [multiGP._info['emulist'][i]._info['ll'] for i in range(multiGP._info['numGPs'])]
    initial_lls = deepcopy(initial_lls)
    
    GPlist = [
        homGP().mle(X=X,Z=Y[:,i],covtype=COVTYPE,maxit=50,
                    settings=SETTINGS)
        for i in range(Y.shape[1])
    ]
    GPlist_initial = deepcopy(GPlist)
    Xnew = X.mean(axis=0).reshape(-1,X.shape[1])
    mpreds = multiGP.predict(x=Xnew)
    Ynew = mpreds._info['mean']
    multiGP.update(x=Xnew,Y=Ynew)
    em_keys = ['ll','theta','beta0']
    for GP in GPlist:
        Ynew = GP.predict(Xnew)['mean']
        GP.update(Xnew,Ynew,maxit=50)
    for i in range(multiGP._info['numGPs']):
        GP = GPlist[i]
        for key in em_keys:
            assert np.allclose(multiGP._info['emulist'][i]._info[key],GP[key])
    # also check that likelihoods changed after update
    for i in range(len(initial_lls)):
        old_ll = initial_lls[i]
        assert not np.allclose(old_ll,multiGP._info['emulist'][i]._info['ll'])

if __name__ == "__main__":
    test_update_kriging_believer()