import numpy as np
from hetgpy import hetGP

###############################################################################
## Heterogeneous GP with all options for the fit
###############################################################################

## ' log-likelihood in the anisotropic case - one lengthscale by variable
## ' Model: K = nu2 * (C + Lambda) = nu using all observations using the replicates information
## ' nu2 is replaced by its plugin estimator in the likelihood
## ' @param X0 unique designs
## ' @param Z0 averaged observations
## ' @param Z replicated observations (sorted with respect to X0)
## ' @param mult number of replicates at each Xi
## ' @param Delta vector of nuggets corresponding to each X0i or pXi, that are smoothed to give Lambda
## ' @param logN should exponentiated variance be used
## ' @param SiNK should the smoothing come from the SiNK predictor instead of the kriging one
## ' @param theta scale parameter for the mean process, either one value (isotropic) or a vector (anistropic)
## ' @param k_theta_g constant used for linking nuggets lengthscale to mean process lengthscale, i.e., theta_g[k] = k_theta_g * theta[k], alternatively theta_g can be used
## ' @param theta_g either one value (isotropic) or a vector (anistropic), alternative to using k_theta_g
## ' @param g nugget of the nugget process
## ' @param pX matrix of pseudo inputs locations of the noise process for Delta (could be replaced by a vector to avoid double loop)
## ' @param beta0 mean, if not provided, the MLE estimator is used
## ' @param eps minimal value of elements of Lambda
## ' @param covtype covariance kernel type
## ' @param penalty should a penalty term on Delta be used?
## ' @param hom_ll reference homoskedastic likelihood
## ' @export

def fit(fitinfo, x, theta, f, 
        lower=None, 
        upper=None,
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
        Z vector of all observations. If using a list with ``X``, ``Z`` has to be ordered with respect to ``X0``, and of length ``sum(mult)``
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

    f = f.flatten()
    model = hetGP()
    model.mle(X = x, 
            Z = f, 
            known=known,
            noiseControl=noiseControl, 
            lower=lower, 
            upper=upper,
            settings=settings, 
            init=init, 
            eps=eps,
            covtype=covtype
    )
    fitinfo['ll']   = model.get('ll')
    fitinfo['Delta'] = model.get('Delta')
    fitinfo['theta'] = model.get('theta')
    fitinfo['g'] = model.get('g')  
    fitinfo['k_theta_g'] = model.get('k_theta_g') 
    fitinfo['theta_g'] = model.get('theta_g')  
    fitinfo['nmean'] = model.get('nmean')
    fitinfo['Lambda'] = model.get('Lambda')
    fitinfo['logN'] = model.get('logN')
    fitinfo['nu_hat_var'] = model.get('nu_hat_var')
    fitinfo['nu_hat'] = model.get('nu_hat')
    fitinfo['Kgi']    = model.get('Kgi')
    fitinfo['SiNK']   = model.get('SiNK')
    fitinfo['Ki'] = model.get('Ki') 
    fitinfo['X0'] = model.get('X0')
    fitinfo['Z0'] = model.get('Z0')
    fitinfo['Z']  = model.get('Z')
    fitinfo['mult'] = model.get('mult')
    fitinfo['covtype'] = model.get('covtype')
    fitinfo['beta0'] = model.get('beta0')
    fitinfo['eps'] = model.get('eps')
    fitinfo['trendtype'] = model.get('trendtype')
    fitinfo['is_homGP'] = model.get('is_homGP')
    return
    
class hetGPWrapper(hetGP):
    '''
    A class that converts the information in fitinfo (from the fit and predict methods) to a class so
    it can be used to make predictions with hetgpy.hetGP

    '''
    def __init__(self,fitinfo):
        
        keys_to_transfer = [
            'X0','Z0','Z', # data
            'covtype',     # kernel
            'theta', 'g', 'beta0','trendtype', # hyperparameters
            'Lambda', 'Delta', # latent noise
            'theta_g','k_theta_g', # noise hyperparameters
            'nu_hat_var', # noise variance
            'Kgi', # noise covariance inverse
            'SiNK',
            'nmean',
            'logN', 
            'g', # nugget
            'Ki', # inverse covariance matrix
            'eps', # for numeric stability
            'nu_hat' # output from maximum likelihood
        ]

        # model hyperparameters
        for key in keys_to_transfer:
            self[key] = fitinfo[key]   

def predict(predinfo, fitinfo, x, theta, thetaprime=None,rep_no=None, **kwargs):
    r'''
    Wrapper method for hetgpy.hetGP.predict
    '''
    GP = fitinfo.get('model')
    if GP is None:
        # use wrapper class to instantiate trained GP
        GP = hetGPWrapper(fitinfo=fitinfo)


    preds = GP.predict(x=x,xprime=thetaprime)
    # ensure naming consistency
    predinfo['mean']   = preds['mean']
    predinfo['var']    = preds['sd2']
    predinfo['nugs']   = preds['nugs']
    predinfo['covmat'] = preds['cov']
    return