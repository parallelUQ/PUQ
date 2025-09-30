'''
Set of functions to fit a heteroskedastic GP to each dimension of output data
'''
import numpy as np
from PUQ.surrogate import emulator

def fit(fitinfo, x, theta, f, 
        lower=None, 
        upper=None,
        maxit=100,
        noiseControl={'k_theta_g_bounds': (1, 100), 'g_max': 1e2, 'g_bounds': (1e-6, 1)}, 
        init={}, 
        known={}, 
        eps=np.sqrt(np.finfo(float).eps),
        settings={"linkThetas": 'joint', "logN": True, "initStrategy": 'residuals', 
                  "checkHom": True, "penalty": True, "trace": 0, "return.matrices": True, 
                  "return.hom": False, "factr": 1e9}, 
        covtype = 'Gaussian', **kwargs):
    r'''
    Wrapper function for hetgpy.hetGP.mleHetGP

    Arguments
    ---------
    fitinfo: dictionary/class holding emulator results
    x : ndarray_like
            matrix of all designs, one per row, or list with elements:
            - ``X0`` matrix of unique design locations, one point per row
            - ``Z0`` vector of averaged observations, of length ``len(X0)``
            - ``mult`` number of replicates at designs in ``X0``, of length ``len(X0)``
    theta: not used
    f : ndarray_like
        output array for training. One GP is trained for each output column.
    lower,upper : ndarray_like 
        optional bounds for the ``theta`` parameter (see :func: covariance_functions.cov_gen for the exact parameterization).
        In the multivariate case, it is possible to give vectors for bounds (resp. scalars) for anisotropy (resp. isotropy)
    noiseControl : dict
        dict with elements related to optimization of the noise process parameters:
            - ``g_min``, ``g_max`` minimal and maximal noise to signal ratio (of the mean process)
            - ``lowerDelta``, ``upperDelta`` optional vectors (or scalars) of bounds on ``Delta``, of length ``len(X0)`` (default to ``np.repeat(eps, X0.shape[0])`` and ``np.repeat(noiseControl["g_max"], X0.shape[0])`` resp., or their ``log``) 
            - ``lowerpX``, ``upperpX`` optional vectors of bounds of the input domain if `pX` is used.
            - ``lowerTheta_g``, ``upperTheta_g`` optional vectors of bounds for the lengthscales of the noise process if ``linkThetas == 'none'``. Same as for ``theta`` if not provided.
            - ``k_theta_g_bounds`` if ``linkThetas == 'joint'``, vector with minimal and maximal values for ``k_theta_g`` (default to ``(1, 100)``). See Notes.
            - ``g_bounds`` vector for minimal and maximal noise to signal ratios for the noise of the noise process, i.e., the smoothing parameter for the noise process. (default to ``(1e-6, 1)``).
    settings : dict 
            dict for options about the general modeling procedure, with elements:
                - ``linkThetas`` defines the relation between lengthscales of the mean and noise processes. Either ``'none'``, ``'joint'``(default) or ``'constr'``, see Notes.
                - ``logN``, when ``True`` (default), the log-noise process is modeled.
                - ``initStrategy`` one of ``'simple'``, ``'residuals'`` (default) and ``'smoothed'`` to obtain starting values for ``Delta``, see Notes
                - ``penalty``  when ``True``, the penalized version of the likelihood is used (i.e., the sum of the log-likelihoods of the mean and variance processes, see References).
                - ``hardpenalty`` is ``True``, the log-likelihood from the noise GP is taken into account only if negative (default if ``maxit > 1000``).
                - ``checkHom`` when ``True``, if the log-likelihood with a homoskedastic model is better, then return it.
                - ``trace`` optional scalar (default to ``0``). If negative, fit silently. If ``0``, only high level information is given. If ``1``, information is given about the result of the heterogeneous model optimization. Level ``2`` gives more details. Level ``3`` additionaly displays all details about initialization of hyperparameters.
                - ``return_matrices`` boolean to include the inverse covariance matrix in the object for further use (e.g., prediction).
                - ``return_hom`` boolean to include homoskedastic GP models used for initialization (i.e., ``modHom`` and ``modNugs``).
                - ``factr`` (default to 1e9) and ``pgtol`` are available to be passed to `options` for L-BFGS-B in :func: ``scipy.optimize.minimize``.   
    eps : float
        jitter used in the inversion of the covariance matrix for numerical stability
    init,known :  dict
        optional lists of starting values for mle optimization or that should not be optimized over, respectively.
        Values in ``known`` are not modified, while it can happen to these of ``init``, see Notes. 
        One can set one or several of the following:
            - ``theta`` lengthscale parameter(s) for the mean process either one value (isotropic) or a vector (anistropic)
            - ``Delta`` vector of nuggets corresponding to each design in ``X0``, that are smoothed to give ``Lambda`` (as the global covariance matrix depends on ``Delta`` and ``nu_hat``, it is recommended to also pass values for ``theta``)
            - ``beta0`` constant trend of the mean process
            - ``k_theta_g`` constant used for link mean and noise processes lengthscales, when ``settings['linkThetas'] == 'joint'``
            - ``theta_g`` either one value (isotropic) or a vector (anistropic) for lengthscale parameter(s) of the noise process, when ``settings['linkThetas'] != 'joint'``
            - ``g`` scalar nugget of the noise process
            - ``g_H`` scalar homoskedastic nugget for the initialisation with a :func: homGP.mleHomGP. See Notes.
            - ``pX`` matrix of fixed pseudo inputs locations of the noise process corresponding to Delta
    covtype : str 
            covariance kernel type, either ``'Gaussian'``, ``'Matern5_2'`` or ``'Matern3_2'``, see :func: ``~covariance_functions.cov_gen``
    maxit : int
            maximum number of iterations for `L-BFGS-B` of :func: ``scipy.optimize.minimize`` dedicated to maximum likelihood optimization
    '''
    numGPs = f.shape[1]
    emulist = [dict() for x in range(0, numGPs)]
    for i in range(numGPs):
        emu = emulator(
                x=x,
                theta=np.array([[0]]),
                f=f[:,i:i+1],
                method="hetGP",
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
    fitinfo["f"] = f.T
    fitinfo["theta"] = x
    fitinfo["emulist"] = emulist
    fitinfo["numGPs"] = numGPs
    return
def predict(predinfo,fitinfo,x,theta,thetaprime,**kws):
    r'''
    Wrapper method for hetGP.predict

    Parameters
    ----------
    predinfo: dict
        (empty) dictionary that will hold prediction results
    fitinfo: dict
        dictionary with hetgpy.hetGP-trained hyperparameters and inverse covariance matrices. fitinfo is converted back into a hetgpy.hetGP object for prediction
    x: ndarray
        nxd numpy array for prediction. Must match same number of columns as supplied to `fitinfo["X0"]`
    theta: ndarray
        Deprecated, but used to specify output dimension
    thetaprime: ndarray
        nxd numpy array for calculating covariance matrix
    kwargs: dict
        additional keyword arguments passed to hetgpy.hetGP.predict
    
    Returns
    -------
    None, but predinfo is populated with `mean`, `variance`,`nugs`, and `covmat` fields. 
    `mean`, `variance`, and `nugs` are stored as ndarrays, with each column corresponding to the prediction for the ith output column. 
    `covmat` is stored as a list of matrices with the ith element corresponding to the ith output column
    '''
    numGPs = fitinfo['numGPs']
    emulist = [dict() for x in range(0, numGPs)]

    # instantiate outputs
    nr, nc = (x.shape[0],numGPs)
    for key in ('mean','var','nugs'):
        predinfo[key] = np.zeros(shape=(nc,nr),dtype=float)

    if thetaprime is None:
        predinfo['covmat'] = np.zeros(shape=(nc,nr,nr),dtype=float)
    else:
        predinfo['covmat'] = np.zeros(shape=(nc,nr,thetaprime.shape[0]),dtype=float) 
                         
    for i in range(numGPs):
        preds = fitinfo['emulist'][i].predict(x=x,thetaprime=thetaprime)
        predinfo['mean'][i,:] = preds._info['mean']
        predinfo['var'][i,:] = preds._info['var']
        predinfo['nugs'][i,:] = preds._info['nugs']
        predinfo['covmat'][i,:,:] = preds._info['covmat']


    predinfo["S"] = np.full((numGPs, numGPs, x.shape[0]), np.nan)
    predinfo["R"] = np.full((numGPs, numGPs, x.shape[0]), np.nan)
    for i in range(0, x.shape[0]):
        C = np.diag(predinfo['var'][:,i])
        R = np.diag(predinfo['nugs'][:,i])
        predinfo["S"][:, :, i] = C
        predinfo["R"][:, :, i] = R
        
    return

def update(fitinfo, x,Y = None,**kwargs):
    r'''
    Update function for hetGP

    Parameters
    ----------
    fitinfo: dictionary that contains the fit information for a hetgpy.hetGP object
    x: array of new design locations
    Y: new response. If None, then a kriging believer approach is used to impute the predicted mean at the design location
    kwargs: key-value pairs that get passed to hetgpy.hetGP.update. 
            Must be one of: ginit, lower, upper, noiseControl, settings, known, maxit
    
    Returns
    -------
    None, but fitinfo is updated in place and individual emulator are accessed via fitinfo["emulist"]
    '''
    numGPs = fitinfo['numGPs']
    for i in range(numGPs):
        emu = fitinfo['emulist'][i]
        if Y is not None:
            Yi = Y[:,i]
        else:
            Yi = None
        emu.update(x=x,Y=Yi,**kwargs)
    return