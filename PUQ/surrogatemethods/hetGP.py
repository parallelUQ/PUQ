import numpy as np
from scipy.linalg import cholesky, inv
from PUQ.surrogatemethods.covariances import cov_gen, partial_cov_gen
import scipy.optimize as spo
from PUQ.surrogatemethods.helpers import auto_bounds, find_reps
from PUQ.surrogatemethods.homGP import fit as fitHomGP
from PUQ.surrogatemethods.homGP import predict as predictHomGP

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
    model = mleHetGP(X = theta, 
                     Z = f, 
                     known=known,
                     noiseControl=noiseControl, 
                     lower=lower, 
                     upper=upper,
                     settings=settings, 
                     init=init, 
                     eps=eps,
                     covtype=covtype)
    
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
    fitinfo['Ki'] = model.get('Ki') 
    fitinfo['X0'] = model.get('X0')
    fitinfo['Z0'] = model.get('Z0')
    fitinfo['mult'] = model.get('mult')
    fitinfo['beta0'] = model.get('beta0')
    fitinfo['eps'] = model.get('eps')
    fitinfo['trendtype'] = model.get('trendtype')
    fitinfo['is_homGP'] = model.get('is_homGP')
    return
    
    
def mleHetGP(X, Z, 
             lower=None, 
             upper=None, 
             known={}, 
             noiseControl={},
             init={}, 
             covtype="Gaussian",
             maxit=100, 
             eps=np.sqrt(np.finfo(float).eps),
             settings={}):
    

    elem = find_reps(X, Z, return_Zlist=False)

    X0 = elem['X0']
    Z0 = elem['Z0']
    Z = elem['Z']
    mult = elem['mult']
    N = len(Z)
    n = len(X0)

    if (lower is None or upper is None):
        auto_thetas = auto_bounds(X=X0, covtype=covtype)
        
        if lower is None:
            lower = auto_thetas["lower"]
        if upper is None:
            upper = auto_thetas["upper"]    
    
    if len(lower) != len(upper):
        print("upper and lower should have the same size")
        sys.exit(1)
    

    jointThetas, constrThetas = False, False
    
    # Check if known['theta_g'] is not None
    if known.get('theta_g') is not None:
        settings['linkThetas'] = False
    
    # Check if settings['linkThetas'] is None
    if settings.get('linkThetas') is None:
        jointThetas = True
    else:
        if settings['linkThetas'] == 'joint':
            jointThetas = True
        elif settings['linkThetas'] == 'constr':
            constrThetas = True
    
    logN = True
    if settings.get('logN') is not None:
        logN = settings['logN']

    if settings.get('return.matrices') is None:
        settings['return.matrices'] = True
    
    if settings.get('return.hom') is None:
        settings['return.hom'] = False
    
    if jointThetas and noiseControl.get('k_theta_g_bounds') is None:
        noiseControl['k_theta_g_bounds'] = [1, 100]
    
    if settings.get('initStrategy') is None:
        settings['initStrategy'] = 'residuals'
    
    if settings.get('factr') is None:
        settings['factr'] = 1e9
    
    penalty = True
    if settings.get('penalty') is not None:
        penalty = settings['penalty']
        
    if settings.get('checkHom') is None:
        settings['checkHom'] = True

    trace = 0
    if settings.get('trace') is not None:
        trace = settings['trace']
    
    components = []
    if known.get('theta') is None:
        components.append('theta')
    else:
        init['theta'] = known['theta']
    
    if known.get('Delta') is None:
        components.append('Delta')
    else:
        init['Delta'] = known['Delta']
    
    if jointThetas:
        if known.get('k_theta_g') is None:
            components.append('k_theta_g')
        else:
            init['k_theta_g'] = known['k_theta_g']
    
    if not jointThetas and known.get('theta_g') is None:
        components.append('theta_g')
    else:
        if not jointThetas:
            init['theta_g'] = known['theta_g']
    
    if known.get('g') is None:
        components.append('g')
    else:
        init['g'] = known['g']
    

    trendtype = 'OK'
    if known.get('beta0') is not None:
        trendtype = 'SK'
        # beta0 = known['beta0']
    
    if noiseControl.get('g_bounds') is None:
        noiseControl['g_bounds'] = [1e-6, 1]
    
    if len(components) == 0 and known.get('theta_g') is None:
        known['theta_g'] = known['k_theta_g'] * known['theta']

    # Automatic Initialization
    modHom, modNugs = None, None
    
    if init.get('theta') is None or init.get('Delta') is None:
        # A) homoskedastic mean process
        if known.get('g_H') is not None:
            g_init = None
        else:
            g_init = init.get('g_H')

            # Initial value for g of the homoskedastic process: based on the mean variance at replicates
            # compared to the variance of Z0
            if any(mult > 5):
                #mean_var_replicates = np.mean((fast_tUY2(mult, (Z - np.tile(Z0, mult))**2)/mult)[np.where(mult > 5)])
                v1 = (Z - np.repeat(Z0, mult))**2
                v2 = np.zeros(len(mult))
                counter = 0
                for mid, m in enumerate(mult):
                    v2[mid] = sum([v1[counter+i] for i in range(m)])
                    counter += m
                mean_var_replicates = np.mean((v2/mult)[mult > 5])

                if g_init is None:
                    g_init = np.array([mean_var_replicates / np.var(Z0, ddof=1)])
    
                if noiseControl.get('g_max') is None:
                    noiseControl['g_max'] = max(1e2, 100 * g_init)
    
                if noiseControl.get('g_min') is None:
                    noiseControl['g_min'] = eps
            else:
                if g_init is None:
                    g_init = np.array([0.1])
    
                if noiseControl.get('g_max') is None:
                    noiseControl['g_max'] = 1e2
    
                if noiseControl.get('g_min') is None:
                    noiseControl['g_min'] = eps
                    
        rKI = True if settings['checkHom'] else False
  
        fitinfo_homGP = {}
        fitHomGP(fitinfo=fitinfo_homGP,
                 x=np.array([[0]]), 
                 theta={'X0':X0, 'Z0':Z0, 'mult':mult}, 
                 f=Z,          
                 lower=lower,
                 upper=upper,
                 known={'theta':known.get('theta'), 
                        'g':known.get('g_H'), 
                        'beta0':known.get('beta0')},
                 init={'theta':init.get('theta'), 
                       'g':g_init},
                 covtype=covtype,
                 maxit=maxit,
                 noiseControl={'g_bounds':(noiseControl['g_min'], noiseControl['g_max'])},
                 eps=eps,
                 settings={'return.Ki':rKI},
                 method="homGP")
    
        if known.get('theta') is None:
            init["theta"] = fitinfo_homGP["theta"]
            
        if init.get('Delta') is None:
            predinfo_homGP = {}
            predictHomGP(predinfo=predinfo_homGP, 
                                   fitinfo=fitinfo_homGP, 
                                   x=np.array([[0]]), 
                                   theta=X0)
            predHom = predinfo_homGP['mean']

            # squared deviation from the homoskedastic prediction mean to the actual observations
            nugs_est = (np.repeat(predHom, repeats=mult) - Z) ** 2
            nugs_est /= fitinfo_homGP["nu_hat"]
                        
            if logN:
                nugs_est = np.maximum(nugs_est, np.finfo(float).eps)  # avoid problems on deterministic test functions
                nugs_est = np.log(nugs_est)
 
            nugs_est0 = np.zeros(len(mult))
            counter = 0
            for mid, m in enumerate(mult):
                nugs_est0[mid] = sum([nugs_est[counter+i] for i in range(m)])/m
                counter += m
        else:
            nug_est0 = init['Delta']
        
        if constrThetas:
            noiseControl['lowerTheta_g'] = fitinfo_homGP['theta']

        if settings['initStrategy'] == 'simple':
            if logN:
                init['Delta'] = np.repeat(np.log(fitinfo_homGP['g']), X0.shape[0])
            else:
                init['Delta'] = np.repeat(fitinfo_homGP['g'], X0.shape[0])
        
        if settings['initStrategy'] == 'residuals':
            init['Delta'] = nugs_est0

    if ((init.get('theta_g') is None and init.get('k_theta_g') is None) or init.get('g') is None):
        # B) Homogeneous noise process
        if jointThetas:
            if init.get('k_theta_g') is None:
                init['k_theta_g'] = 1
                init['theta_g'] = init['theta']
            else:
                init['theta_g'] = init['k_theta_g'] * init['theta']
        
            if noiseControl.get('lowerTheta_g') is None:
                noiseControl['lowerTheta_g'] = init['theta_g'] - eps
        
            if noiseControl.get('upperTheta_g') is None:
                noiseControl['upperTheta_g'] = init['theta_g'] + eps
        
        if not jointThetas and init.get('theta_g') is None:
            init['theta_g'] = init['theta']
    
        if noiseControl.get('lowerTheta_g') is None:
            noiseControl['lowerTheta_g'] = lower
        
        if noiseControl.get('upperTheta_g') is None:
            noiseControl['upperTheta_g'] = upper
    
        # If a homogeneous process of the mean has already been computed, it is used for estimating the parameters of the noise process
        if 'nugs_est' in locals():
            if init.get('g') is None:
                # mean_var_replicates_nugs = np.mean((fast_tUY2(mult, (nugs_est - np.repeat(nugs_est0, mult)))**2) / mult)
                v1 = (nugs_est - np.repeat(nugs_est0, mult))**2
                v2 = np.zeros(len(mult))
                counter = 0
                for mid, m in enumerate(mult):
                    v2[mid] = sum([v1[counter+i] for i in range(m)])
                    counter += m
                mean_var_replicates_nugs = np.mean((v2/mult))
                init['g'] = np.array([mean_var_replicates_nugs / np.var(nugs_est0, ddof=1)])

            fitnugs_homGP = {}
            fitHomGP(fitinfo=fitnugs_homGP,
                     x=np.array([[0]]), 
                     theta={'X0': X0, 'Z0': nugs_est0, 'mult': mult}, 
                     f=nugs_est,
                     lower=noiseControl['lowerTheta_g'], 
                     upper=noiseControl['upperTheta_g'],
                     init={'theta': init['theta_g'], 'g': init['g']}, 
                     covtype=covtype, 
                     noiseControl=noiseControl,
                     maxit=maxit, 
                     eps=eps, 
                     settings={'return.Ki': False})
            prednugs_homGP = {}
            predictHomGP(predinfo=prednugs_homGP, 
                         fitinfo=fitnugs_homGP, 
                         x=np.array([[0]]), 
                         theta=X0)
            # prednugs = predinfo_homGP['mean']
        
        else:
            if 'nugs_est0' not in locals():
                nugs_est0 = init['Delta']
        
            if init['g'] is None:
                init['g'] = np.array([0.05])
        
            fitnugs_homGP = {}
            fitHomGP(fitinfo=fitnugs_homGP,
                     x=np.array([[0]]), 
                     theta={'X0': X0, 'Z0': nugs_est0, 'mult': np.repeat(1, X0.shape[0])}, 
                     f=nugs_est0,
                     lower=noiseControl['lowerTheta_g'], 
                     upper=noiseControl['upperTheta_g'],
                     init={'theta': init['theta_g'], 'g': init['g']}, 
                     covtype=covtype, 
                     noiseControl=noiseControl,
                     maxit=maxit, 
                     eps=eps, 
                     settings={'return.Ki': False})
            prednugs_homGP = {}
            predictHomGP(predinfo=prednugs_homGP, 
                         fitinfo=fitnugs_homGP, 
                         x=np.array([[0]]), 
                         theta=X0)
            # prednugs = predinfo_homGP['mean']
            
        if settings['initStrategy'] == 'smoothed':
            init['Delta'] = prednugs_homGP['mean']

        if known.get('g') is None:
            init['g'] = fitnugs_homGP['g']

        if jointThetas and init.get('k_theta_g') is None:
            init['k_theta_g'] = 1
    
        if not jointThetas and init.get('theta_g') is None:
            init['theta_g'] = fitnugs_homGP['theta']
    
    if noiseControl.get('lowerTheta_g') is None:
        noiseControl['lowerTheta_g'] = lower

    if noiseControl.get('upperTheta_g') is None:
        noiseControl['upperTheta_g'] = upper

    def fn(par, X0, Z0, Z, mult, Delta=None, theta=None, g=None, k_theta_g=None, 
           theta_g=None, logN=False, beta0=None, hom_ll=None, env=None):
        
        idx = 0  # to store the first non-used element of par
    
        if theta is None:
            idx = len(init['theta'])
            theta = par[:idx]

        if Delta is None:
            Delta = par[idx:idx + len(init['Delta'])]
            idx += len(init['Delta'])
               
        if jointThetas and k_theta_g is None:
            k_theta_g = par[idx]
            idx += 1

        if not jointThetas and theta_g is None:
            theta_g = par[idx:idx + len(init['theta_g'])]
            idx += len(init_theta_g)
            
        if g is None:
            g = par[idx]
            idx += 1

        loglik = logLikHet(X0=X0, Z0=Z0, Z=Z, 
                           mult=mult, 
                           Delta=Delta, 
                           theta=theta, 
                           g=g, 
                           k_theta_g=k_theta_g, 
                           theta_g=theta_g, 
                           logN=logN, 
                           beta0=beta0, 
                           hom_ll=hom_ll, 
                           env=env, 
                           eps=eps, 
                           penalty=penalty, 
                           trace=trace)

        # Update env with the maximum log-likelihood and corresponding parameters
        if env is not None and not np.isnan(loglik):
            if env.get('max_loglik') is None or loglik > env['max_loglik']:
                env['max_loglik'] = loglik
                env['arg_max'] = par
        return -1*loglik

    def gr(par, X0, Z0, Z, mult, Delta=None, theta=None, g=None, k_theta_g=None, 
           theta_g=None, logN=False, beta0=None, hom_ll=None, env=None):
        
        idx = 0  # to store the first non-used element of par
    
        if theta is None:
            theta = par[idx:idx + len(init['theta'])]
            idx += len(init['theta'])
            
        if Delta is None:
            Delta = par[idx:idx + len(init['Delta'])]
            idx += len(init['Delta'])
            
        if jointThetas and k_theta_g is None:
            k_theta_g = par[idx]
            idx += 1
            
        if not jointThetas and theta_g is None:
            theta_g = par[idx:idx + len(init['theta_g'])]
            idx += len(init['theta_g'])
            
        if g is None:
            g = par[idx]
            idx += 1

        dloglik = dlogLikHet(X0=X0, Z0=Z0, Z=Z, 
                             mult=mult, 
                             Delta=Delta, 
                             theta=theta, 
                             g=g, 
                             k_theta_g=k_theta_g, 
                             theta_g=theta_g,
                             logN=logN, 
                             beta0=beta0, 
                             hom_ll=hom_ll,
                             env=env,
                             eps=eps,
                             penalty=penalty, 
                             components=components)

        negdloglik = [-l for l in dloglik]
        return negdloglik


    parinit, lowerOpt, upperOpt = [], [], []

    if trace > 2:
        print("Initial value of the parameters:")
    
    if known.get("theta") is None:
        parinit.extend(init["theta"])
        lowerOpt.extend(lower)
        upperOpt.extend(upper)
        if trace > 2:
            print("Theta:", init["theta"])
    
    if known.get("Delta") is None:
        if noiseControl.get("lowerDelta") is None or len(noiseControl["lowerDelta"]) != n:
            if logN:
                noiseControl["lowerDelta"] = np.array([np.log(eps)]) # [np.log(eps)] * n
            else:
                noiseControl["lowerDelta"] = np.array([eps]) # [eps] * n
    
        if len(noiseControl["lowerDelta"]) == 1:
            noiseControl["lowerDelta"] = np.repeat(noiseControl["lowerDelta"], n) #[noiseControl["lowerDelta"][0]] * n
    
        if noiseControl.get("g_max") is None:
            noiseControl["g_max"] = 1e2
    
        if noiseControl.get("upperDelta") is None or len(noiseControl["upperDelta"]) != n:
            if logN:
                noiseControl["upperDelta"] = np.array([np.log(noiseControl["g_max"])]) # [np.log(noiseControl["g_max"])]
            else:
                noiseControl["upperDelta"] = np.array([noiseControl["g_max"]]) # [noiseControl["g_max"]]
    
        if len(noiseControl["upperDelta"]) == 1:
            noiseControl["upperDelta"] = np.repeat(noiseControl["upperDelta"], n) # [noiseControl["upperDelta"][0]] * n
    
        lowerOpt.extend(noiseControl["lowerDelta"])
        upperOpt.extend(noiseControl["upperDelta"])
        parinit.extend(init["Delta"])
        if trace > 2:
            print("Delta:", init["Delta"])
            
    if jointThetas and known.get("k_theta_g") is None:
        parinit.extend([init["k_theta_g"]])
        lowerOpt.extend([noiseControl["k_theta_g_bounds"][0]])
        upperOpt.extend([noiseControl["k_theta_g_bounds"][1]])
        if trace > 2:
            print("k_theta_g:", init["k_theta_g"])

    if not jointThetas and known.get("theta_g") is None:
        parinit.extend(init["theta_g"])
        lowerOpt.extend([noiseControl["lowerTheta_g"]])
        upperOpt.extend([noiseControl["upperTheta_g"]])
        if trace > 2:
            print("theta_g:", init["theta_g"])
    
    if known.get("g") is None:
        parinit.extend([init["g"]])
        if trace > 2:
            print("g:", init["g"])
    
        lowerOpt.extend([noiseControl["g_bounds"][0]])
        upperOpt.extend([noiseControl["g_bounds"][1]])
    
    mle_par = known.copy()  # Store inferred and known parameters

    if components is not None:
        if fitinfo_homGP is not None:
            hom_ll = fitinfo_homGP['ll']
        else:
            # Compute reference homoskedastic likelihood, with fixed theta for speed
            modHom_tmp = mleHomGP(X={"X0": X0, "Z0": Z0, "mult": mult},
                                  Z=Z,
                                  lower=lower,
                                  upper=upper,
                                  init={"theta": known["theta"], "g": known["g_H"], "beta0": known["beta0"]},
                                  covtype=covtype,
                                  noiseControl={"g_bounds": [noiseControl["g_min"], noiseControl["g_max"]]},
                                  eps=eps,
                                  settings={"return.Ki": False})
    
            hom_ll = modHom_tmp.ll

        try:
            # print("Starting")
            envtemp = {}
            opval = spo.minimize(fn,
                                 parinit,
                                 method='L-BFGS-B',
                                 options={'gtol': 0.01, 'maxiter': 100},
                                 jac=gr,
                                 bounds=spo.Bounds(lowerOpt, upperOpt),
                                 args=(X0, Z0, Z, mult, known.get('Delta'), known.get('theta'), 
                                       known.get('g'), known.get('k_theta_g'), known.get('theta_g'),
                                       logN, known.get('beta0'), hom_ll, envtemp))
            
            out = 1*opval.x
            out_ll = -1*opval.fun 
        except:
            print('use best value so far')
            out = envtemp["arg_max"]
            out_ll = envtemp["max_loglik"]

        if trace > 1:
            print("Name | Value | Lower bound | Upper bound")
        
        # Post-processing
        idx = 0
        if known.get("theta") is None:
            mle_par["theta"] = np.array(out[:len(init["theta"])])
            idx += len(init["theta"])
            if trace > 1:
                print(f"Theta | {mle_par['theta']} | {lower} | {upper}")
        
        if known.get("Delta") is None:
            mle_par["Delta"] = np.array(out[idx : (idx + len(init["Delta"]))])
            idx += len(init["Delta"])
            if trace > 1:
                for ii in range(1, (len(mle_par["Delta"]) // 5) + 2):
                    i_tmp = np.arange(5 * (ii - 1), min(5 * ii, len(mle_par["Delta"])))
                    if logN:
                        print(
                            f"Delta | {np.array(mle_par['Delta'])[i_tmp]} | {np.maximum(np.log(eps * mult[i_tmp]), init['Delta'][i_tmp] - np.log(1000))} | {init['Delta'][i_tmp] + np.log(100)}"
                        )
                    if not logN:
                        print(
                            f"Delta | {np.array(mle_par['Delta'])[i_tmp]} | {np.maximum(mult[i_tmp] * eps, init['Delta'][i_tmp] / 1000)} | {init['Delta'][i_tmp] * 100}"
                        )
        if jointThetas:
            if known.get('k_theta_g') is None:
                mle_par['k_theta_g'] = np.array(out[idx])
                idx += 1
            mle_par['theta_g'] = mle_par['k_theta_g'] * mle_par['theta']
            
            if trace > 1:
                print(f"k_theta_g | {mle_par['k_theta_g']} | {noiseControl['k_theta_g_bounds'][0]} | {noiseControl['k_theta_g_bounds'][1]}")
        
        if not jointThetas and known.get("theta_g") is None:
            mle_par['theta_g'] = np.array(out[idx : (idx + len(init["theta_g"]))])
            idx += len(init["theta_g"])
            if trace > 1:
                print(f"theta_g | {mle_par['theta_g']} | {noiseControl['lowerTheta_g']} | {noiseControl['upperTheta_g']}")
            
        if known.get('g') is None:
            mle_par['g'] = np.array(out[idx])
            idx += 1
            if trace > 1:
                print(f"g | {mle_par['g']} | {noiseControl['g_bounds'][0]} | {noiseControl['g_bounds'][1]}")
        
    else:
        out = {"message": "All hyperparameters given, no optimization \n",
               "count": 0,
               "value": None
               }
    
    if penalty:
        ll_non_pen = logLikHet(X0=X0, Z0=Z0, Z=Z, mult=mult, 
                               Delta=mle_par['Delta'], 
                               theta=mle_par['theta'], 
                               g=mle_par['g'], 
                               k_theta_g=mle_par['k_theta_g'], 
                               theta_g=mle_par['theta_g'], 
                               logN=logN, 
                               beta0=mle_par.get('beta0'), 
                               hom_ll=None, 
                               env=None,
                               eps=eps, 
                               penalty=False, 
                               trace=trace)

    else:
        ll_non_pen = 1*out_ll#-1*opval.fun 
    

    if fitinfo_homGP is not None:
        if fitinfo_homGP['ll'] >= ll_non_pen:
            if trace >= 1:
                print(f"Homoskedastic model has higher log-likelihood: {fitinfo_homGP['ll']} compared to {ll_non_pen}")
     
            if settings['checkHom']:
                if trace >= 1:
                    print("Return homoskedastic model")
                fitinfo_homGP['is_homGP'] = True
                return fitinfo_homGP


    Cg = cov_gen(X1=X0, theta=mle_par['theta_g'])
    Kg_c = cholesky(Cg + np.diag(eps + mle_par['g'] / mult))
    Kgi = np.linalg.inv((Kg_c))
    Kgi = Kgi @ Kgi.T

    nmean = np.sum(Kgi @ mle_par['Delta']) / np.sum(Kgi) ## ordinary kriging mean
    nu_hat_var = max(eps, np.dot((mle_par['Delta'] - nmean), Kgi @ (mle_par['Delta'] - nmean)) / len(mle_par['Delta']))
    M = np.dot(Cg, np.dot(Kgi, mle_par['Delta'] - nmean))
            
    Lambda = nmean + M
    if logN:
        Lambda = np.exp(Lambda)
    else:
        Lambda[Lambda <= 0] = eps

    LambdaN = np.repeat(Lambda, mult)
    C = cov_gen(X1=X0, theta=mle_par['theta'])
    Ki = cholesky(C + np.diag(Lambda / mult + eps))
    Ki = np.linalg.inv((Ki))
    Ki = Ki @ Ki.T
    
    if known.get('beta0') is None:
        mle_par['beta0'] = np.sum(Ki @ Z0) / np.sum(Ki)
    
    psi_0 = np.dot(Z0 - mle_par['beta0'], Ki @ (Z0 - mle_par['beta0']))
    
    nu2 = 1 / N * (np.dot((Z - mle_par['beta0']) / LambdaN, Z - mle_par['beta0']) - np.dot((Z0 - mle_par['beta0']) * mult / Lambda, Z0 - mle_par['beta0']) + psi_0)

    fitinfo = {}
    fitinfo['Delta'] = mle_par['Delta']
    fitinfo['theta'] = mle_par['theta']
    fitinfo['g'] = mle_par['g']  
    fitinfo['k_theta_g'] = mle_par['k_theta_g']  
    fitinfo['theta_g'] = mle_par['theta_g']  
    fitinfo['nmean'] = nmean
    fitinfo['Lambda'] = Lambda
    fitinfo['logN'] = logN
    fitinfo['nu_hat_var'] = nu_hat_var
    fitinfo['nu_hat'] = nu2
    fitinfo['Ki'] = Ki  
    fitinfo['Kgi'] = Kgi  
    fitinfo['X0'] = X0
    fitinfo['Z0'] = Z0
    fitinfo['mult'] = mult
    fitinfo['beta0'] = mle_par['beta0']
    fitinfo['eps'] = eps
    fitinfo['trendtype'] = trendtype
    fitinfo['is_homGP'] = False
    #fitinfo['nll'] = opval.fun
    


    # values = [0.8367, 1.1516, -7.9251, -5.8644, -8.5293, -7.3552, -7.5042,
    #           -7.0089, -8.3106, -8.3518, -5.5814, -8.4839, -8.0848, -8.2968,
    #           -8.4812, -8.7633, -4.8743, 1.0261, 0
    #           ]
    # par = np.array(values)
        
    # fn(par, X0, Z0, Z, mult, hom_ll=-np.inf, env=None, logN=True)
    # gr(par, X0, Z0, Z, mult, Delta=None, theta=None, g=None, k_theta_g=None, theta_g=None, logN=True, beta0=None, hom_ll=-np.inf, env=None)

        
    return fitinfo

def logLikHet(X0, Z0, Z, mult, Delta, theta, g, k_theta_g=None, theta_g=None, 
              logN=False, beta0=None, hom_ll=None, env=None, 
              eps=np.sqrt(np.finfo(float).eps), penalty=True, trace=0):
    n = len(X0)
    N = len(Z)

    if theta_g is None:
        theta_g = k_theta_g * theta

    Cg = cov_gen(X1=X0, theta=theta_g)
    Kg_c = cholesky(Cg + np.diag(eps + g / mult))
    Kgi = np.linalg.inv((Kg_c))
    Kgi = Kgi @ Kgi.T
    nmean = np.sum(Kgi @ Delta) / np.sum(Kgi)
    M = Cg @ (Kgi @ (Delta - nmean))
    Lambda = nmean + M

    if logN:
        Lambda = np.exp(Lambda)
    else:
        Lambda[Lambda <= 0] = eps

    LambdaN = np.repeat(Lambda, mult)

    C = cov_gen(X1=X0, theta=theta)
    if env is not None:
        env['C'] = C
    Ki = cholesky(C + np.diag(Lambda / mult + eps))
    ldetKi = -2 * np.sum(np.log(np.diag(Ki)))
    Ki = np.linalg.inv((Ki))
    Ki = Ki @ Ki.T

    if env is not None:
        env['Cg'] = Cg
        env['Kg_c'] = Kg_c
        env['Kgi'] = Kgi
        env['ldetKi'] = ldetKi
        env['Ki'] = Ki

    if beta0 is None:
        beta0 = np.sum(Ki @ Z0) / np.sum(Ki)

    psi_0 = np.dot(Z0 - beta0, Ki @ (Z0 - beta0))

    psi = 1 / N * (np.dot((Z - beta0) / LambdaN, Z - beta0) - np.dot((Z0 - beta0) * mult / Lambda, Z0 - beta0) + psi_0)

    loglik = -N / 2 * np.log(2 * np.pi) - N / 2 * np.log(psi) + 1 / 2 * ldetKi - 1 / 2 * np.sum(
        (mult - 1) * np.log(Lambda) + np.log(mult)) - N / 2

    if penalty:
        nu_hat_var = np.dot((Delta - nmean), Kgi @ (Delta - nmean)) / len(Delta)
        if nu_hat_var < eps:
            return loglik
        pen = -n / 2 * np.log(nu_hat_var) - np.sum(np.log(np.diag(Kg_c))) - n / 2 * np.log(2 * np.pi) - n / 2

        if hom_ll is not None and loglik < hom_ll and pen > 0:
            if trace > 0:
                print("Penalty is deactivated when unpenalized likelihood is lower than its homGP equivalent")
            return loglik
        
        return loglik + pen

    return loglik

def dlogLikHet(X0, Z0, Z, mult, Delta, theta, g, k_theta_g=None, theta_g=None, 
               logN=True, beta0=None, hom_ll=None, env=None, 
               eps=np.sqrt(np.finfo(float).eps), penalty=True, components=None):
  
    # Verifications
    if k_theta_g is None and theta_g is None:
        print("Either k_theta_g or theta_g must be provided.")
        return None
    
    # Initializations
    if components is None:
        components = ["theta", "Delta", "g"]
        if k_theta_g is None:
            components.append("theta_g")
        else:
            components.append("k_theta_g")
    
    if theta_g is None:
        theta_g = k_theta_g * theta
        
    n = X0.shape[0]
    N = len(Z)
    
    if env is not None:
        Cg = env['Cg']
        Kg_c = env['Kg_c']
        Kgi = env['Kgi']
        M = (Kgi * (-eps - g / mult) + np.eye(n)).T
    else:
        Cg = cov_gen(X1=X0, theta=theta_g)
        Kg_c = cholesky(Cg + np.diag(eps + g / mult))
        Kgi = np.linalg.inv((Kg_c))
        Kgi = Kgi @ Kgi.T
        M = (Kgi * (-eps - g / mult) + np.eye(n)).T
    
    # Precomputations for reuse
    rSKgi = np.sum(Kgi, axis=1)
    sKgi = np.sum(Kgi)
    nmean = np.dot(rSKgi, Delta) / sKgi  # ordinary kriging mean

    # Precomputations for reuse
    KgiD = np.dot(Kgi, (Delta - nmean))
    Lambda = nmean + np.dot(M, (Delta - nmean))
    
    if logN:
        Lambda = np.exp(Lambda)
    else:
        Lambda[Lambda <= 0] = eps
    LambdaN = np.repeat(Lambda, repeats=mult)
    
    if env is not None:
        C = env['C']
        Ki = env['Ki']
        ldetKi = env['ldetKi']
    else:
        C = cov_gen(X1=X0, theta=theta)
        Ki = cholesky(C + np.diag(Lambda / mult + eps))
        ldetKi = -2 * np.sum(np.log(np.diag(Ki)))  # log determinant from Cholesky
        Ki = np.linalg.inv((Ki))
        Ki = Ki @ Ki.T
        
    if beta0 is None:
        beta0 = np.dot(np.sum(Ki, axis=0), Z0) / np.sum(Ki)
    
    # Precomputations for reuse
    KiZ0 = np.dot(Ki, Z0 - beta0)
    rsM = np.sum(M, axis=1)
    
    psi_0 = np.dot(KiZ0, Z0 - beta0)
    psi = np.dot((Z - beta0) / LambdaN, Z - beta0) - np.dot((Z0 - beta0) * mult / Lambda, Z0 - beta0) + psi_0

    if penalty:
        nu_hat_var = np.dot(KgiD, Delta - nmean)/ len(Delta)
        # To prevent numerical issues when Delta = nmean, resulting in divisions by zero
        if nu_hat_var < eps:
            penalty = False
        else:
            loglik = -N/2 * np.log(2*np.pi) - N/2 * np.log(psi/N) + 1/2 * ldetKi - 1/2 * np.sum((mult - 1) * np.log(Lambda) + np.log(mult)) - N/2
            pen = -n/2 * np.log(nu_hat_var) - np.sum(np.log(np.diag(Kg_c))) - n/2 * np.log(2*np.pi) - n/2

            if loglik < hom_ll and pen > 0:
                penalty = False
                

    dLogL_dtheta, dLogL_dDelta, dLogL_dkthetag, dLogL_dthetag, dLogL_dg, dLogL_dpX = None, None, None, None, None, None
    # First component, derivative of logL with respect to theta
    if "theta" in components:
        dLogL_dtheta = np.full(len(theta), np.nan)
        for i in range(len(theta)):
            if len(theta) == 1:
                dC_dthetak = partial_cov_gen(X1=X0, theta=theta, arg="theta_k") * C
            else:
                dC_dthetak = partial_cov_gen(X1=X0[:, i][:, None], theta=theta[i], arg="theta_k") * C

            if "k_theta_g" in components:     

                if len(theta) == 1:
                    dCg_dthetak = partial_cov_gen(X1=X0, theta=k_theta_g * theta, arg="theta_k") * k_theta_g * Cg
                else:
                    dCg_dthetak = partial_cov_gen(X1=X0[:, i][:, None], theta=k_theta_g * theta[i], arg="theta_k") * k_theta_g * Cg
            
                # Derivative Lambda / theta_k (first part)
                dLdtk = dCg_dthetak @ KgiD - M @ (dCg_dthetak @ KgiD)
       
                # (second part)
                dLdtk -= (1 - rsM) * (np.dot(np.dot(rSKgi, dCg_dthetak), np.dot(Kgi, Delta)) * sKgi - np.dot(rSKgi, Delta) * np.dot(np.dot(rSKgi, dCg_dthetak), rSKgi)) / sKgi**2

                if logN:
                    dLdtk *= Lambda

                dK_dthetak = np.add(dC_dthetak, np.diag(dLdtk) / mult)  # dK/dtheta[k]

                term1 = N/2 * (np.dot(((Z - beta0) / LambdaN) * np.repeat(dLdtk, mult), (Z - beta0) / LambdaN) - np.dot(((Z0 - beta0) / Lambda * mult * dLdtk), (Z0 - beta0) / Lambda) +
                               np.dot(np.dot(KiZ0, dK_dthetak), KiZ0)) / psi - 0.5 * np.trace(np.dot(Ki.T, dK_dthetak))

                term2 = 0.5 * np.sum((mult - 1) * dLdtk / Lambda)
                dLogL_dtheta[i] = term1 - term2

                if penalty:
                    dLogL_dtheta[i] += 0.5 * np.dot(np.dot(KgiD, dCg_dthetak), KgiD) / nu_hat_var - 0.5 * np.trace(np.dot(Kgi.T, dCg_dthetak))
            else:
                # N/2 * crossprod(KiZ0, dC_dthetak) %*% KiZ0/psi
                term1 = (N/2) * np.dot(np.dot(KiZ0, dC_dthetak), KiZ0) / psi
                # 1/2 * trace_sym(Ki, dC_dthetak)
                term2 = 0.5 * np.trace(np.dot(Ki.T, dC_dthetak))
                dLogL_dtheta[i] = term1 - term2
    

    # Derivative of logL with respect to Lambda
    if any(x in ["Delta", "g", "k_theta_g", "theta_g"] for x in components):
        v1 = (Z - beta0) ** 2
        v2 = np.zeros(len(mult))
        counter = 0
        for mid, m in enumerate(mult):
            v2[mid] = sum([v1[counter+i] for i in range(m)])
            counter += m
   
        dLogLdLambda = (N / 2 * ((v2 - (Z0 - beta0) ** 2 * mult) / Lambda ** 2
                + KiZ0 ** 2 / mult) / psi - (mult - 1) / (2 * Lambda) - 1 / (2 * mult) * np.diag(Ki))

        if logN:
            dLogLdLambda = Lambda * dLogLdLambda

    # Derivative of Lambda with respect to Delta
    if "Delta" in components:
        dLogL_dDelta = np.dot(M.T, dLogLdLambda) + rSKgi / sKgi * np.sum(dLogLdLambda) - rSKgi / sKgi * np.sum(np.dot(M.T, dLogLdLambda))  # chain rule
        
    # Derivative Lambda / k_theta_g
    if "k_theta_g" in components:
        dCg_dk = partial_cov_gen(X1=X0, theta=theta, k_theta_g=k_theta_g, arg="k_theta_g") * Cg

        dLogL_dkthetag = np.dot(dCg_dk, KgiD) - np.dot(M, np.dot(dCg_dk, KgiD)) - \
                         (1 - rsM) * np.squeeze(
            (rSKgi @ np.dot(dCg_dk, np.dot(Kgi, Delta)) * sKgi - rSKgi @ Delta * (rSKgi @ np.dot(dCg_dk, rSKgi))) / sKgi**2)       
                                 
        dLogL_dkthetag = dLogL_dkthetag @ dLogLdLambda  # chain rule

    # Derivative Lambda / theta_g
    if "theta_g" in components:
        dLogL_dthetag = np.full(len(theta_g), np.nan)
        for i in range(len(theta_g)):
            if len(theta_g) == 1:
                dCg_dthetagk = partial_cov_gen(X1=X0, theta=theta_g, arg="theta_k") * Cg
            else:
                dCg_dthetagk = partial_cov_gen(X1=X0[:, i], theta=theta_g[i], arg="theta_k") * Cg
        
            dLogL_dthetag[i] = np.dot((dCg_dthetagk @ KgiD - M @ (dCg_dthetagk @ KgiD) - 
                                       ((1 - rsM) * (rSKgi @ (dCg_dthetagk @ Kgi @ Delta) * sKgi - 
                                                     rSKgi @ Delta * (rSKgi @ (dCg_dthetagk @ rSKgi.T) @ rSKgi))))/ sKgi**2, dLogLdLambda)

            # Penalty term
            if penalty:
                dLogL_dthetag[i] += 1/2 * np.dot(np.dot(KgiD.T, dCg_dthetagk) @ KgiD, 1/nu_hat_var) - 0.5*np.trace(np.dot(Kgi, dCg_dthetagk)) #trace_sym(Kgi, dCg_dthetagk)/2

    ## Derivative Lambda / g
    if "g" in components:
        dLogL_dg = np.dot(-M @ (KgiD/mult) -
                          (1 - rsM) * (np.dot(Delta, (Kgi @ (rSKgi/mult)) * sKgi) - rSKgi @ Delta * np.sum(rSKgi**2/mult))/sKgi**2,
                          dLogLdLambda)  # chain rule


    # Additional penalty terms on Delta
    if penalty:
        if "Delta" in components:
            dLogL_dDelta = dLogL_dDelta - KgiD / nu_hat_var
        if "k_theta_g" in components:
            dLogL_dkthetag = dLogL_dkthetag + 1/2 * np.dot(KgiD, dCg_dk) @ KgiD / nu_hat_var - 0.5*np.trace(np.dot(Kgi, dCg_dk)) #trace_sym(Kgi, dCg_dk)/2
        if "g" in components:
            dLogL_dg = dLogL_dg + 1/2 * np.dot(KgiD/mult, KgiD) / nu_hat_var - np.sum(np.diag(Kgi)/mult)/2
    
    nested_list = []
    for der in [dLogL_dtheta, dLogL_dDelta, dLogL_dkthetag, dLogL_dthetag, dLogL_dg, dLogL_dpX]:
        if der is not None:
            if der.ndim < 1:
                nested_list.extend(np.array([der]))
            else:
                nested_list.extend(der)
    return nested_list #np.concatenate([dLogL_dtheta, dLogL_dDelta, dLogL_dkthetag, dLogL_dthetag, dLogL_dg, dLogL_dpX])


def predict(predinfo, fitinfo, x, theta, thetaprime=None, rep_no=None, **kwargs):

    if fitinfo.get('Delta') is None:
        return predictHomGP(predinfo, fitinfo, x, theta, thetaprime=thetaprime, rep_no=rep_no)
    else:
        return predicthetGP(predinfo, fitinfo, x, theta, thetaprime=thetaprime, noise_var=False, nugs_only=False, rep_no=rep_no)
        
        
def predicthetGP(predinfo, fitinfo, x, theta, thetaprime=None, noise_var=False, nugs_only=False, rep_no=None, **kwargs):
    
    if rep_no is not None:
        mult = 1*rep_no
    else:
        mult = 1*fitinfo['mult']
    
    if fitinfo.get('Kgi') is None:
        Cg = cov_gen(X1=fitinfo['X0'], theta=fitinfo['theta_g'])
        Kg_c = cholesky(Cg + np.diag(fitinfo['eps'] + fitinfo['g'] / mult))
        Kgi = np.linalg.inv((Kg_c))
        Kgi = Kgi @ Kgi.T
    
    kg = cov_gen(X1=theta, X2=fitinfo['X0'], theta=fitinfo['theta_g'])

    if fitinfo['Ki'] is None:
        C = cov_gen(X1=fitinfo['X0'], theta=fitinfo['theta'])
        Ki = cholesky(C + np.diag(fitinfo['Lambda'] / mult + fitinfo['eps']))
        Ki = np.linalg.inv((Ki))
        fitinfo['Ki'] = Ki @ Ki.T
        
    M = np.dot(kg, np.dot(Kgi, fitinfo['Delta'] - fitinfo['nmean']))

    if fitinfo['logN']:
        nugs = fitinfo['nu_hat'] * np.exp(fitinfo['nmean'] + M)
    else:
        nugs = fitinfo['nu_hat'] * np.maximum(0, fitinfo['nmean'] + M)
        
    if nugs_only:
        return {'nugs': nugs}

    if noise_var:
        if fitinfo['nu_hat_var'] is None:
            fitinfo['nu_hat_var'] = max(fitinfo['eps'], np.dot(np.dot(fitinfo['Delta'] - fitinfo['nmean'], Kgi), fitinfo['Delta'] - fitinfo['nmean'])/len(fitinfo['Delta']))  # To avoid 0 variance
        
        # Assuming kg, object['Kgi'] are NumPy arrays
        term1 = 1 - np.diag(np.dot(kg, np.dot(Kgi.T, kg)))
        term2 = (1 - np.dot(np.sum(Kgi, axis=0), kg))**2 / np.sum(Kgi)

        sd2var = fitinfo['nu_hat'] * fitinfo['nu_hat_var'] * (term1 - term2)
    else:
        sd2var = None
    
    # theta.shape[0] x fitinfo['X0'].shape[0]
    kx = cov_gen(X1=theta, X2=fitinfo['X0'], theta=fitinfo['theta'])
    if fitinfo['trendtype'] == 'SK':
        sd2 = fitinfo['nu_hat'] * (1 - np.diag(np.dot(kx, np.dot(fitinfo['Ki'], kx.T))))
    else:
        sd2 = fitinfo['nu_hat'] * (1 - np.diag(np.dot(kx, np.dot(fitinfo['Ki'], kx.T))) + (1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kx.T ).flatten()**2/np.sum(fitinfo['Ki']) )
    
    if any(sd2 < 0):
        sd2 = np.maximum(0, sd2)
        
    predinfo['mean'] = (fitinfo['beta0'] + np.dot(kx, np.dot(fitinfo['Ki'], fitinfo['Z0'] - fitinfo['beta0'])))[None, :]   
    predinfo['var'] = sd2[None, :]   
    predinfo['sdvar'] = sd2var   
    predinfo['nugs'] = nugs[None, :]  

    if thetaprime is not None:
        # fitinfo['X0'].shape[0] x thetaprime.shape[0]
        kxprime = cov_gen(X1=fitinfo['X0'], X2=thetaprime, theta=fitinfo['theta'])
        if fitinfo['trendtype'] == 'SK':
            if theta.shape[0] < thetaprime.shape[0]:
                cov = fitinfo['nu_hat'] * (cov_gen(X1=theta, X2=thetaprime, theta=fitinfo['theta']) - kx @ fitinfo['Ki'] @ kxprime)
            else:
                cov = fitinfo['nu_hat'] * (cov_gen(X1=theta, X2=thetaprime, theta=fitinfo['theta']) - kx @ (fitinfo['Ki'] @ kxprime))
        else:
            if theta.shape[0] < thetaprime.shape[0]:
                
                cov = fitinfo['nu_hat'] * (cov_gen(X1=theta, X2=thetaprime, theta=fitinfo['theta']) - kx @ fitinfo['Ki'] @ kxprime + \
                                           ((1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kx.T).T @ (1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kxprime))/np.sum(fitinfo['Ki']))
                                           #np.dot(1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kx.T, 1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kxprime)/np.sum(fitinfo['Ki']))
                                           #((1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kx.T).T @ (1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kxprime))/np.sum(fitinfo['Ki']))

            else:
                cov = fitinfo['nu_hat'] * (cov_gen(X1=theta, X2=thetaprime, theta=fitinfo['theta']) - kx @ (fitinfo['Ki'] @ kxprime) + \
                                           np.dot((1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kx.T).T, 1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kxprime)/np.sum(fitinfo['Ki']))
                                           #((1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kx.T).T @ (1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kxprime))/np.sum(fitinfo['Ki']))
                                           #np.dot((1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kx.T).T, 1 - np.sum(fitinfo['Ki'], axis=0)[None, :] @ kxprime)/np.sum(fitinfo['Ki']))
    
        predinfo['covmat'] = cov
        
    return