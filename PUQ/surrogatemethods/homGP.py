import sys
import numpy as np
from scipy.linalg import cholesky, inv
from PUQ.surrogatemethods.covariances import cov_gen, partial_cov_gen
import scipy.optimize as spo
from PUQ.surrogatemethods.helpers import auto_bounds, find_reps

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

    f = f.flatten()
    model = mleHomGP(X = theta, 
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
    fitinfo['mult'] = model['mult']
    fitinfo['beta0'] = model['beta0']
    fitinfo['ll'] = model['ll']    
    fitinfo['trendtype'] = model['trendtype']  
    fitinfo['eps'] = model['eps']  
    fitinfo['is_homGP'] = True
    return

def mleHomGP(X, Z, lower=None, upper=None, known={}, noiseControl={},
             init={}, covtype="Gaussian",
             maxit=100, eps=np.sqrt(np.finfo(float).eps),
             settings={}):


    if isinstance(X, dict):
        X0 = np.array(X.get('X0'))
        Z0 = np.array(X.get('Z0'))
        mult = np.array(X.get('mult'))

        if np.sum(mult) != len(Z):
            raise ValueError("Length(Z) should be equal to sum(mult)")

        if X0.ndim == 1:
            X0 = X0[:, np.newaxis]

        if len(Z0) != X0.shape[0]:
            raise ValueError("Dimension mismatch between Z0 and X0")
    
    else:
        elem = find_reps(X, Z, return_Zlist=False)
        X0 = elem['X0']
        Z0 = elem['Z0']
        Z = elem['Z']
        mult = elem['mult']
        
    N = len(Z)

    if (lower is None or upper is None):
        auto_thetas = auto_bounds(X=X0, covtype=covtype)
        if lower is None:
            lower = auto_thetas["lower"]
        if upper is None:
            upper = auto_thetas["upper"]     
        
        if known.get("theta") is None and init.get("theta") is None:
            init['theta'] = np.sqrt(upper * lower)
    
    if len(lower) != len(upper):
        print("upper and lower should have the same size")
        sys.exit(1)
    
    if settings.get('return.Ki') is None:
        settings['return.Ki'] = True
        
    if noiseControl.get("g_bounds") is None:
        noiseControl["g_bounds"] = [np.sqrt(np.finfo(float).eps), 1e2]

    g_min = np.array([noiseControl["g_bounds"][0]])
    g_max = np.array([noiseControl["g_bounds"][1]])
    
    beta0 = known.get("beta0")
    
    if (known.get("theta") is None and init.get("theta") is None):
        init["theta"] = 0.9*lower + 0.1*upper
     
    if(known.get('g') is None and init.get('g') is None):
        if(any(mult > 2)):
            v1 = (Z - np.repeat(Z0, mult)) ** 2
            v2 = np.zeros(len(mult))
            counter = 0
            for mid, m in enumerate(mult):
                v2[mid] = sum([v1[counter+i] for i in range(m)])
                counter += m
            init['g'] = np.array([np.mean((v2/mult)[mult > 2])/np.var(Z0, ddof=1)])
        else:
            init['g'] = np.array([0.1])

    trendtype = 'OK'
    if known.get("beta0") is not None:
        trendtype = 'SK'

    def fn(par, X0, Z0, Z, mult, beta0, theta, g, env):

        idx = 0  # to store the first non-used element of par
        
        # If theta is not provided, extract it from par
        if theta is None:
            theta = par[:len(init['theta'])]
            idx += len(init['theta'])
        
        # If g is not provided, extract it from par
        if g is None:
            g = par[idx]
        
        # Compute the log-likelihood using logLikHom function
        loglik = logLikHom(X0, Z0, Z, mult, theta, g, beta0, covtype, eps, env)

        # Update env with the maximum log-likelihood and corresponding parameters
        if env is not None and not np.isnan(loglik):
            if env.get('max_loglik') is None or loglik > env['max_loglik']:
                env['max_loglik'] = loglik
                env['arg_max'] = par

        return -loglik
    
    def gr(par, X0, Z0, Z, mult, beta0, theta, g, env):
        idx = 0
        components = []
    
        if theta is None:
            theta = par[:len(init['theta'])]
            idx += len(init['theta'])
            components.append("theta")
    
        if g is None:
            g = par[idx]
            components.append("g")
    
        gr_val = dlogLikHom(X0=X0, Z0=Z0, Z=Z, mult=mult, theta=theta, g=g, beta0=beta0, covtype=covtype, eps=eps, components=components, env=env)
        
        return -gr_val

    envtemp = {}
    if known.get("g") is not None and known.get("theta") is not None:
        theta_out = known['theta']
        g_out = known['g']
        out_ll = logLikHom(X0, Z0, Z, mult, theta_out, g_out, beta0, covtype, eps, None)
    else:
        parinit, lowerOpt, upperOpt = None, None, None
        if known.get("theta") is None:
            parinit = init['theta']
            lowerOpt = lower
            upperOpt = upper
        if known.get("g") is None:

            parinit = np.concatenate((parinit, init['g']))
            lowerOpt = np.concatenate((lowerOpt, g_min))
            upperOpt = np.concatenate((upperOpt, g_max))
        try:
            opval = spo.minimize(fn,
                                 parinit,
                                 method='L-BFGS-B',
                                 #options={'gtol': 0.001},
                                 jac=gr,
                                 bounds=spo.Bounds(lowerOpt, upperOpt),
                                 args=(X0, Z0, Z, mult, known.get("theta"), known.get("g"), beta0, envtemp))
            
            out = opval.x
            out_ll = -1*opval.fun
        except:
            print("use best value so far")
            out = envtemp["arg_max"]
            out_ll = envtemp["max_loglik"]
        
        g_out = out[-1] if known.get("g") is None else known["g"]
        theta_out = out[0:-1] if known.get("theta") is None else known["theta"]
        
    # Temporarily store Cholesky transform of K in Ki
    C_out = cov_gen(X1=X0, theta=theta_out)
    Ki = cholesky(C_out + np.diag(eps + g_out / mult))
    Ki = np.linalg.inv((Ki))
    Ki = Ki @ Ki.T

    if beta0 is None:
        beta0 = np.sum(np.dot(Ki, Z0)) / np.sum(Ki)
 
    psi_0 = np.dot(np.dot(Z0 - beta0, Ki), Z0 - beta0)
    nu = 1/N * ((np.dot(Z - beta0, Z - beta0) - np.dot((Z0 - beta0) * mult, Z0 - beta0)) / g_out + psi_0)

    fitinfo = {}
    fitinfo['theta'] = theta_out
    fitinfo['g'] = g_out  
    fitinfo['nu_hat'] = nu
    fitinfo['Ki'] = Ki  
    fitinfo['X0'] = X0
    fitinfo['Z0'] = Z0
    fitinfo['beta0'] = beta0
    fitinfo['ll'] = out_ll
    fitinfo['mult'] = mult
    fitinfo['trendtype'] = trendtype
    fitinfo['eps'] = eps
    
    # print(logLikHom(X0, Z0, Z, mult, np.array([0.01951545099]), np.array([0.07451224610]), None, covtype, eps, None))
    # print(dlogLikHom(X0, Z0, Z, mult, np.array([0.01951545099]), np.array([0.07451224610]), beta0=None, covtype="Gaussian",
    #                 eps=np.sqrt(np.finfo(float).eps), components=["theta", "g"], env=None))
    
    return fitinfo
                   
# @return loglikelihood value
  
def logLikHom(X0, Z0, Z, mult, theta, g, beta0=None, covtype="Gaussian", eps=np.sqrt(np.finfo(float).eps), env=None):
    n = X0.shape[0]
    N = len(Z)

    # Temporarily store Cholesky transform of K in Ki
    C = cov_gen(X1=X0, theta=theta)

    if env is not None:
        env['C'] = C
        
    Ki = cholesky(C + np.diag(eps + g / mult))

    ldetKi = -2 * np.sum(np.log(np.diag(Ki)))  # log determinant from Cholesky
    Ki = np.linalg.inv((Ki))
    Ki = Ki @ Ki.T

    if env is not None:
        env['Ki'] = Ki
        
    if beta0 is None:
        #beta0 = np.sum(np.dot(Ki, Z0)) / np.sum(Ki)
        beta0 = np.sum(Ki, axis=0) @ Z0 / np.sum(Ki)
    
    psi_0 = np.dot(np.dot(Z0 - beta0, Ki), Z0 - beta0)

    psi = 1/N * ((np.dot(Z - beta0, Z - beta0) - np.dot((Z0 - beta0) * mult, Z0 - beta0)) / g + psi_0)
    
    loglik = -N/2 * np.log(2*np.pi) - N/2 * np.log(psi) + 1/2 * ldetKi - (N - n)/2 * np.log(g) - 1/2 * np.sum(np.log(mult)) - N/2

    return loglik

def dlogLikHom(X0, Z0, Z, mult, theta, g, beta0=None, covtype="Gaussian",
                eps=np.sqrt(np.finfo(float).eps), components=["theta", "g"], env=None):

    k = len(Z)
    n = X0.shape[0]

    if env is not None:
        C = env['C']
        Ki = env['Ki']
    else:
        C = cov_gen(X1=X0, theta=theta)
        Ki = cholesky(C + np.diag(eps + g / mult))
        Ki = np.linalg.inv((Ki))
        Ki = Ki @ Ki.T

    if beta0 is None:
        beta0 = np.sum(Ki, axis=0) @ Z0 / np.sum(Ki)
        #beta0 = np.sum(np.dot(Ki, Z0)) / np.sum(Ki)

    
    Z0 = Z0 - beta0
    Z = Z - beta0

    KiZ0 = np.dot(Ki, Z0)  # to avoid recomputing

    psi = np.dot(Z0, KiZ0)

    tmp1, tmp2 = None, None

    # First component, derivative with respect to theta
    if "theta" in components:
        tmp1 = np.full(len(theta), np.nan)

        if len(theta) == 1:
            dC_dthetak = partial_cov_gen(X1=X0, theta=theta, type=covtype, arg="theta_k") * C
            tmp1[0] = k / 2 * np.dot(np.dot(KiZ0, dC_dthetak), KiZ0) / (
                    (np.dot(Z, Z) - np.dot(Z0 * mult, Z0)) / g + psi) - 1 / 2 * np.trace(Ki@dC_dthetak)
        else:
        
            for i in range(len(theta)):
                dC_dthetak = partial_cov_gen(X1=X0[:, i][:, None], theta=theta[i], type=covtype, arg="theta_k") * C
                tmp1[i] = k / 2 * np.dot(np.dot(KiZ0, dC_dthetak), KiZ0) / (
                        (np.dot(Z, Z) - np.dot(Z0 * mult, Z0)) / g + psi) - 1 / 2 * np.trace(Ki@dC_dthetak) #1 / 2 * trace_sym(Ki, dC_dthetak)

    # Second component derivative with respect to g
    if "g" in components:
        tmp2 = np.full(1, np.nan)
        tmp2[0] = k / 2 * ((np.dot(Z, Z) - np.dot(Z0 * mult, Z0)) / g ** 2 + np.sum(KiZ0 ** 2 / mult)) / ((np.dot(Z, Z) - np.dot(Z0 * mult, Z0)) / g + psi) - (k - n) / (2 * g) - 1 / 2 * np.sum(np.diag(Ki) / mult)
    
    return np.concatenate([tmp1, tmp2])


def predict(predinfo, fitinfo, x, theta, thetaprime=None, **kwargs):
    
    if fitinfo['Ki'] is None:
        C = cov_gen(X1=fitinfo['X0'], theta=fitinfo['theta'])
        Ki = cholesky(C + np.diag(fitinfo['eps'] + fitinfo['g'] / fitinfo['mult']))
        Ki = np.linalg.inv((Ki))
        fitinfo['Ki'] = Ki @ Ki.T
        
    fitinfo['Ki_scaled'] = fitinfo['Ki']/fitinfo['nu_hat']
    
    kx = fitinfo['nu_hat'] * cov_gen(X1=theta, X2=fitinfo['X0'], theta=fitinfo['theta'])

    mean = fitinfo['beta0'] + kx @ (fitinfo['Ki_scaled'] @ (fitinfo['Z0'] - fitinfo['beta0']))
    
    if fitinfo['trendtype'] == 'SK':
        var = fitinfo['nu_hat'] - np.diag(kx @ (fitinfo['Ki_scaled'] @ kx.T))
    else:
        var = fitinfo['nu_hat'] - np.diag(kx @ (fitinfo['Ki_scaled'] @ kx.T)) + (1 - np.sum(fitinfo['Ki_scaled'], axis=1) @ kx.T)**2/np.sum(fitinfo['Ki_scaled'])  
    
    if any(var < 0):
        var = np.maximum(0, var)
    predinfo['mean'] = mean[None, :]          
    predinfo['var'] = var[None, :]      
    predinfo['nugs'] = np.repeat(fitinfo['nu_hat']*fitinfo['g'], theta.shape[0])[None, :]  
    
    if thetaprime is not None:
        # fitinfo['X0'].shape[0] x thetaprime.shape[0]
        kxprime = fitinfo['nu_hat'] * cov_gen(X1=fitinfo['X0'], X2=thetaprime, theta=fitinfo['theta'])
        if fitinfo['trendtype'] == 'SK':
            if theta.shape[0] < thetaprime.shape[0]:
                cov = fitinfo['nu_hat'] * cov_gen(X1=theta, X2=thetaprime, theta=fitinfo['theta']) - kx @ fitinfo['Ki_scaled'] @ kxprime
            else:
                cov = fitinfo['nu_hat'] * cov_gen(X1=theta, X2=thetaprime, theta=fitinfo['theta']) - kx @ (fitinfo['Ki_scaled'] @ kxprime)
        else:

            if theta.shape[0] < thetaprime.shape[0]:
                cov = fitinfo['nu_hat'] * cov_gen(X1=theta, X2=thetaprime, theta=fitinfo['theta']) - kx @ fitinfo['Ki_scaled'] @ kxprime + \
                    ((1 - np.sum(fitinfo['Ki_scaled'], axis=0)[None, :] @ kx.T).T @ (1 - np.sum(fitinfo['Ki_scaled'], axis=0)[None, :] @ kxprime))/np.sum(fitinfo['Ki_scaled'])
                                           #np.dot(1 - np.sum(fitinfo['Ki_scaled'], axis=0)[None, :] @ kx.T, 1 - np.sum(fitinfo['Ki_scaled'], axis=0)[None, :] @ kxprime)/np.sum(fitinfo['Ki_scaled'])
            else:
                cov = fitinfo['nu_hat'] * cov_gen(X1=theta, X2=thetaprime, theta=fitinfo['theta']) - kx @ (fitinfo['Ki_scaled'] @ kxprime) + \
                                           np.dot((1 - np.sum(fitinfo['Ki_scaled'], axis=0)[None, :] @ kx.T).T, 1 - np.sum(fitinfo['Ki_scaled'], axis=0)[None, :] @ kxprime)/np.sum(fitinfo['Ki_scaled'])
    
        predinfo['covmat'] = cov
    return 

