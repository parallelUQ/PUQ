import sys
sys.path.append('./')
import numpy as np
from scipy.stats import qmc
from PUQ.surrogate import emulator
from hetgpy import homGP
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

if __name__ == "__main__":
    test_predict()