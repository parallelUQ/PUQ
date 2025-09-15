import numpy as np
from hetgpy import homGP
from hetgpy.auto_bounds import auto_bounds
from hetgpy.find_reps import find_reps

################################################################################
## Homoskedastic noise
################################################################################

## Model: noisy observations with unknown homoskedastic noise
## K = nu^2 * (C + g * I)
# X0 unique designs matrix
# Z0 averaged observations at X0
# Z observations vector (all observations)
# mult number of replicates at each unique design
# theta vector of lengthscale hyperparameters (or one for isotropy)
# g noise variance for the process
# beta0 trend

def fit(fitinfo, x, theta, f, lower=None, upper=None,
        known={}, 
        noiseControl={"g_bounds": [np.sqrt(np.finfo(float).eps), 100]}, 
        init={}, 
        covtype='Gaussian', 
        maxit=100, 
        eps=np.sqrt(np.finfo(float).eps), 
        settings={"return.Ki": True, "factr": 1e7}, 
        **kwargs):
    r'''
    Wrapper function for hetgpy.homGP.mleHomGP

    Arguments
    ---------
    fitinfo: dictionary/class of results
    x: nxd design matrix (must have at least one column)
    f: output array for training
    theta: not used
    lower,upper : ndarray_like 
            optional bounds for the ``theta`` parameter (see :func: covariance_functions.cov_gen for the exact parameterization).
            In the multivariate case, it is possible to give vectors for bounds (resp. scalars) for anisotropy (resp. isotropy)
        noiseControl : dict
            dict with element:
                - ``g_bounds`` vector providing minimal and maximal noise to signal ratio (default to ``(sqrt(MACHINE_DOUBLE_EPS), 100)``).
        settings : dict 
                dict for options about the general modeling procedure, with elements:
                    - ``return_Ki`` boolean to include the inverse covariance matrix in the object for further use (e.g., prediction).
                    - ``factr`` (default to 1e7) and ``pgtol`` are available to be passed to `options` for L-BFGS-B in :func: ``scipy.optimize.minimize``.   
        eps : float
            jitter used in the inversion of the covariance matrix for numerical stability
        known : dict
            optional dict of known parameters (e.g. ``beta0``, ``theta``, ``g``)
        init :  dict
            optional lists of starting values for mle optimization:
                - ``theta_init`` initial value of the theta parameters to be optimized over (default to 10% of the range determined with ``lower`` and ``upper``)
                - ``g_init`` vector of nugget parameter to be optimized over
        covtype : str 
                covariance kernel type, either ``'Gaussian'``, ``'Matern5_2'`` or ``'Matern3_2'``, see :func: ``~covariance_functions.cov_gen``
        maxit : int
                maximum number of iterations for `L-BFGS-B` of :func: ``scipy.optimize.minimize`` dedicated to maximum likelihood optimization
    


    Returns
    -------
    fitinfo: dictionary containing fit results

    '''
    f = f.flatten()
    model = homGP()
    model.mleHomGP(X = x, 
                     Z = f, 
                     lower=lower, 
                     upper=upper,
                     known=known,
                     noiseControl=noiseControl, 
                     init=init,
                     covtype=covtype,
                     maxit=maxit,
                     eps=eps,
                     settings=settings)
    
    fitinfo['theta'] = model['theta']
    fitinfo['g'] = model['g']  
    fitinfo['nu_hat'] = model['nu_hat']
    fitinfo['Ki'] = model['Ki']  
    fitinfo['X0'] = model['X0']
    fitinfo['Z0'] = model['Z0']
    fitinfo['Z']  = model['Z']
    fitinfo['mult'] = model['mult']
    fitinfo['beta0'] = model['beta0']
    fitinfo['ll'] = model['ll']
    fitinfo['covtype'] = model['covtype']
    fitinfo['nu_hat']  = model['nu_hat']    
    fitinfo['trendtype'] = model['trendtype']  
    fitinfo['eps'] = model['eps']  
    fitinfo['is_homGP'] = True
    return


class homGPWrapper(homGP):
    '''
    A class that converts the information in fitinfo (from the fit and predict methods) to a class so
    it can be used to make predictions with hetgpy.homGP

    '''
    def __init__(self,fitinfo):
        
        keys_to_transfer = [
            'X0','Z0','Z', # data
            'covtype',     # kernel
            'theta', 'g', 'beta0','trendtype', # hyperparameters
            'Ki', # inverse covariance matrix
            'eps', # for numeric stability
            'nu_hat' # output from maximum likelihood
        ]

        # model hyperparameters
        for key in keys_to_transfer:
            self[key] = fitinfo[key]


def predict(predinfo, fitinfo, x, theta, thetaprime=None, **kwargs):
    r'''
    Wrapper method for hetgpy.homGP.predict
    '''
    GP = fitinfo.get('model')
    if GP is None:
        # use wrapper class to instantiate trained GP
        GP = homGPWrapper(fitinfo=fitinfo)


    preds = GP.predict(x=x,xprime=thetaprime)
    # ensure naming consistency
    predinfo['mean']   = preds['mean']
    predinfo['var']    = preds['sd2']
    predinfo['nugs']   = preds['nugs']
    predinfo['covmat'] = preds['cov']
