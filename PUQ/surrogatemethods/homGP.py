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
    
    for key in model.__dict__.keys():
        fitinfo[key] = model.get(key) 
    fitinfo['is_homGP'] = True
    del model
    return


class homGPWrapper(homGP):
    '''
    A class that converts the information in fitinfo (from the fit and predict methods) to a class so
    it can be used to make predictions with hetgpy.homGP

    '''
    def __init__(self,fitinfo):
        # model hyperparameters
        for key in fitinfo.keys():
            setattr(self,key,fitinfo[key])


def predict(predinfo, fitinfo, x, theta, thetaprime=None, **kwargs):
    r'''
    Wrapper method for hetgpy.homGP.predict
    '''
    GP = fitinfo.get('model')
    if GP is None:
        # use wrapper class to instantiate trained GP
        GP = homGPWrapper(fitinfo=fitinfo)
    # handle kws
    kws = {}
    eligible_keys = ['nugs_only','interval','interval_lower','interval_upper']
    for key in eligible_keys:
        if key in kwargs.keys():
            kws[key] = kwargs.get('nugs_only')


    preds = GP.predict(x=x,xprime=thetaprime,**kws)
    # ensure naming consistency
    predinfo['mean']   = preds.get('mean')
    predinfo['var']    = preds.get('sd2')
    predinfo['nugs']   = preds.get('nugs')
    predinfo['covmat'] = preds.get('cov')
    del GP

def update(fitinfo, x,Y = None,**kwargs):
    r'''
    Update function for homGP

    Parameters
    ----------
    fitinfo: dictionary that contains the fit information for a hetgpy.homGP object
    x: array of new design locations
    Y: new response. If None, then 
    kwargs: key-value pairs that get passed to hetgpy.homGP.update. Must be one of:
    '''
    # validate kwargs
    valid_kws = ('ginit','lower','upper',
                 'noiseControl','settings','known','maxit')
    for kw in kwargs.keys():
        if kw not in valid_kws:
            raise ValueError(f"{kw} not found, must be one of {valid_kws}")
    GP = homGPWrapper(fitinfo)
    if Y is None:
        maxit = 0 # impute mean response and do not update hyperparams
        Y = GP.predict(x)['mean']
    else:
        maxit = kwargs.get('maxit',100)
    kwargs['maxit'] = maxit
    GP.update(Xnew=x,Znew=Y,**kwargs)
    for key in GP.__dict__.keys():
        fitinfo[key] = GP.get(key)
    del GP
    return
