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