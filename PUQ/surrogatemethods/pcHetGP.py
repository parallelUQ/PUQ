from PUQ.surrogate import emulator
from PUQ.surrogatemethods.hetGP import predict as predict_hetGP
import numpy as np
from PUQ.surrogatemethods.covariances import cov_gen
from scipy.linalg import cholesky, inv
from PUQ.surrogatemethods.helpers import find_reps

def fit(fitinfo, x, theta, f, 
        lower=None, upper=None,
                noiseControl={'k_theta_g_bounds': (1, 100), 'g_max': 1e2, 'g_bounds': (1e-6, 1)}, 
                init={}, 
                known={}, 
                settings={"linkThetas": 'joint', "logN": True, "initStrategy": 'residuals', 
                          "checkHom": True, "penalty": True, "trace": 0, "return.matrices": True, 
                          "return.hom": False, "factr": 1e9}, 
                covtype = 'Gaussian', 
                pc_settings={'standardize': True, 'latent': False},
                **kwargs):
    
    numGPs = f.shape[0]
    # elements = []
    # for k in range(0, numGPs):
    #     elem = find_reps(theta, f[k, :], return_Zlist=False)
    #     X0 = elem['X0']
    #     Z0 = elem['Z0']
    #     Z = elem['Z']
    #     mult = elem['mult']
    #     elements.append({'X0':X0, 'Z0':Z0, 'Z':Z, 'mult':mult})
    
    h = np.zeros((numGPs, 1))
    s = np.ones((numGPs))
    fs = np.zeros(f.shape)
    
    if pc_settings['standardize']:
        # print("Standardizing")
        fitinfo['standardize'] = True
        for k in range(0, numGPs):
            h[k, 0] = np.mean(f[k, :])
            s[k] = np.std(f[k, :]) 

    G = np.diag(s)
    Gi = np.diag(1/s)
    fs = Gi@(f - h)
    B = np.eye(numGPs)

    if pc_settings['latent']:
        #print("Dimension reduction")
        fitinfo['latent'] = True

        U, S, _ = np.linalg.svd(fs, full_matrices=False)
        exp_var = np.cumsum(S**2)/np.cumsum(S**2)[-1]
        numGPs = int(np.argwhere(np.array(exp_var) > 0.99)[0]) + 1
        
        # d x q orthogonal matrix
        B = U[:, 0:numGPs]

        # trasform q x n
        W = B.T @ fs

    emulist = [dict() for x in range(0, numGPs)]
    for i in range(0, numGPs):
        if pc_settings['latent']:
            print("Latent surrogate")
            fi = W[i, :][None, :]
        else:
            # print("On site surrogate")
            fi = fs[i, :][None, :]
        emu = emulator(x=np.array([[i]]), 
                       theta=theta, 
                       f=fi,                
                       method="hetGP",
                       args={"noiseControl":noiseControl, 
                             "lower":lower, 
                             "upper":upper,
                             "settings":settings, 
                             "init":init, 
                             "known":known,
                             "covtype":covtype})
        
        emulist[i] = emu._info
    
    fitinfo['theta'] = theta
    fitinfo['f'] = f
    fitinfo['emulist'] = emulist
    fitinfo['numGPs'] = numGPs
    fitinfo['fs'] = fs
    fitinfo['h'] = h
    fitinfo['G'] = G
    fitinfo['Gi'] = Gi
    fitinfo['B'] = B

    return

def predict(predinfo, fitinfo, x, theta, thetaprime, **kwargs):
    

    numGPs = fitinfo['numGPs']
    d = fitinfo['fs'].shape[0]
    n = theta.shape[0]
    
    # calculate predictive mean and variance
    mean_pc = np.full((numGPs, n), np.nan)
    var_pc = np.full((numGPs, n), np.nan)
    nugs_pc = np.full((numGPs, n), np.nan)

    if thetaprime is not None: 
        npr = thetaprime.shape[0]
        covmat_pc = np.full((numGPs, n, npr), np.nan)    
    for i in range(0, numGPs):
        predinfo_hetGP = {}
        info = fitinfo['emulist'][i]
        predict_hetGP(predinfo=predinfo_hetGP, 
                               fitinfo=info, 
                               x=np.array([[i]]), 
                               theta=theta,
                               thetaprime=thetaprime)

        mean_pc[i, :] = predinfo_hetGP['mean']
        var_pc[i, :] = predinfo_hetGP['var']
        nugs_pc[i, :] = predinfo_hetGP['nugs']   

        if thetaprime is not None: 
            covmat_pc[i, :, :] = predinfo_hetGP['covmat']  
    
    # calculate predictive mean and variance
    predinfo['mean_o'] = 1*mean_pc
    predinfo['var_o'] = 1*var_pc    
    predinfo['nugs_o'] = 1*nugs_pc  
    if thetaprime is not None: 
        predinfo['cov_o'] = 1*covmat_pc

    predinfo['S'] = np.full((d, d, n), np.nan)
    predinfo['R'] = np.full((d, d, n), np.nan)
    predinfo['var'] = np.full((d, n), np.nan)
    predinfo['nugs'] = np.full((d, n), np.nan)
    predinfo['var_noisy'] = np.full((d, n), np.nan)
    if thetaprime is not None: 
        predinfo['covmat'] = np.full((d, n, npr), np.nan)    
    predinfo['mean'] = fitinfo['h'] + fitinfo['G']@(fitinfo['B']@mean_pc)
    
    for i in range(0, theta.shape[0]):
        C = np.diag(var_pc[:, i])
        R = np.diag(nugs_pc[:, i])
        predinfo['S'][:, :, i] = fitinfo['G']@fitinfo['B']@C@fitinfo['B'].T@fitinfo['G']
        predinfo['R'][:, :, i] = fitinfo['G']@fitinfo['B']@R@fitinfo['B'].T@fitinfo['G']
        predinfo['var'][:, i] = np.diag(predinfo['S'][:, :, i])
        predinfo['nugs'][:, i] = np.diag(predinfo['R'][:, :, i])

    predinfo['var_noisy'] = predinfo['var'] + predinfo['nugs']      
    
    return predinfo


def update(fitinfo, x, X0new=None, mult=None):
    numGPs = fitinfo['numGPs']

    Gi = fitinfo["Gi"]
    h = fitinfo["h"]
    pred_ct = predict(predinfo={}, fitinfo=fitinfo, x=x, theta=X0new, thetaprime=None)
    mu_ct = pred_ct["mean"]
    mu_ct = Gi@(mu_ct - h)
    Z0new = mu_ct.T
    #print(mu_ct.shape)
    
    for i in range(0, numGPs):
        info = fitinfo['emulist'][i]

        if info['is_homGP'] == True:
            info["X0"] = np.concatenate((info["X0"], X0new), axis=0)
            info["Z0"] = np.concatenate((info["Z0"], np.array([Z0new[0, i]])))
            info["mult"] = np.concatenate((info["mult"], np.array([mult])))
            
            C = cov_gen(X1=info['X0'], theta=info['theta'])
            Ki = cholesky(C + np.diag(info['eps'] + info['g'] / info['mult']))
            Ki = np.linalg.inv((Ki))
            info['Ki'] = Ki @ Ki.T
        else:
            Cg = cov_gen(X1=info['X0'], theta=info['theta_g'])
            Kg_c = cholesky(Cg + np.diag(info['eps'] + info['g'] / info["mult"]))
            Kgi = np.linalg.inv((Kg_c))
            Kgi = Kgi @ Kgi.T
            
            kg = cov_gen(X1=X0new, X2=info['X0'], theta=info['theta_g'])        
            M = np.dot(kg, np.dot(Kgi, info['Delta'] - info['nmean']))
            
            info["X0"] = np.concatenate((info["X0"], X0new), axis=0)
            info["Z0"] = np.concatenate((info["Z0"], np.array([Z0new[0, i]])))
            info["mult"] = np.concatenate((info["mult"], np.array([mult])))
            info["Delta"] = np.concatenate((info["Delta"], M + info["nmean"]))
            
            # print(info["Z0"])
            # print(info["Delta"])
            # print(info["Lambda"])
            # Updated
            Cg = cov_gen(X1=info['X0'], theta=info['theta_g'])
            Kg_c = cholesky(Cg + np.diag(info['eps'] + info['g'] / info["mult"]))
            Kgi = np.linalg.inv((Kg_c))
            Kgi = Kgi @ Kgi.T
            M = np.dot(Cg, np.dot(Kgi, info["Delta"] - info["nmean"]))
            Lambda = info["nmean"] + M
            if info["logN"]:
                Lambda = np.exp(Lambda)
            else:
                Lambda[Lambda <= 0] = info["eps"] 
            
            info["Lambda"] = Lambda
            
            C = cov_gen(X1=info['X0'], theta=info['theta'])
            Ki = cholesky(C + np.diag(info['Lambda'] / info["mult"] + info['eps']))
            Ki = np.linalg.inv((Ki))
            info['Ki'] = Ki @ Ki.T


# def update_alloc(fitinfo, x, X0new=None, mult=None):
#     numGPs = fitinfo['numGPs']

#     Gi = fitinfo["Gi"]
#     h = fitinfo["h"]
#     pred_ct = predict(predinfo={}, fitinfo=fitinfo, x=x, theta=X0new, thetaprime=None)
#     mu_ct = pred_ct["mean"]
#     mu_ct = Gi@(mu_ct - h)
#     Z0new = mu_ct.T

#     for i in range(0, numGPs):
#         info = fitinfo['emulist'][i]  
#         for cid, ct in enumerate(X0new):
#              if mult[cid] > 0:
#                  ids = np.all(info["X0"] == ct, axis=1)
#                  #print(info["Z0"][ids])
#                  info["Z0"][ids] = (info["Z0"][ids]*info["mult"][ids] +  Z0new[0, i]*mult[cid])/(info["mult"][ids] + mult[cid])
#                  #print(info["Z0"][ids])
#                  info["mult"][ids] += mult[cid]

#         if info['is_homGP'] == True:
#             C = cov_gen(X1=info['X0'], theta=info['theta'])
#             Ki = cholesky(C + np.diag(info['eps'] + info['g'] / info['mult']))
#             Ki = np.linalg.inv((Ki))
#             info['Ki'] = Ki @ Ki.T
#         else:
#             Cg = cov_gen(X1=info['X0'], theta=info['theta_g'])
#             Kg_c = cholesky(Cg + np.diag(info['eps'] + info['g'] / info["mult"]))
#             Kgi = np.linalg.inv((Kg_c))
#             Kgi = Kgi @ Kgi.T
#             M = np.dot(Cg, np.dot(Kgi, info["Delta"] - info["nmean"]))
#             Lambda = info["nmean"] + M
#             if info["logN"]:
#                 Lambda = np.exp(Lambda)
#             else:
#                 Lambda[Lambda <= 0] = info["eps"] 
            
#             info["Lambda"] = Lambda
            
#             C = cov_gen(X1=info['X0'], theta=info['theta'])
#             Ki = cholesky(C + np.diag(info['Lambda'] / info["mult"] + info['eps']))
#             Ki = np.linalg.inv((Ki))
#             info['Ki'] = Ki @ Ki.T
    
    
    
# def predict_updated(predinfo, emu, x, theta, mult=None, theta_cand=None, thetaprime=None):

#     fitinfo = emu._info
#     q = fitinfo['numGPs']
#     d = len(x)
#     n = theta.shape[0]
#     predinfo['mean_o'] = np.full((q, n), np.nan)
#     predinfo['var_o'] = np.full((q, n), np.nan)
#     if thetaprime is not None: 
#         predinfo['cov_o'] = np.full((q, n, thetaprime.shape[0]), np.nan)       
#     X0 = fitinfo['emulist'][0]['X0']
        
#     if theta_cand is None:
#         theta_train = 1*X0
#         pred_nug = emu.predict(x=x, theta=theta_train)
#     else:
#         theta_train = 1*X0
#         theta_train = np.concatenate((theta_train, theta_cand), axis=0)
#         pred_nug = emu.predict(x=x, theta=theta_train)
#         pred_cand = emu.predict(x=x, theta=theta_cand)
#         mean_cand = pred_cand.mean()
#         mean_cand = fitinfo['B'].T @ (fitinfo['Gi']@(mean_cand - fitinfo['h']))

#     for i in range(0, q):
#         info = fitinfo['emulist'][i]

#         if theta_cand is None:
#             if info['is_homGP']:
#                 Lambda_train = np.repeat(info['g'], theta_train.shape[0])
#             else:
#                 Lambda_train = pred_nug._info['nugs_o'][i, :]/info['nu_hat'] #info['Lambda']
            
#             if mult is not None:
#                 mult_train = info['mult'] + mult
#             else:
#                 mult_train = 1*info['mult']
#             Z0 = info['Z0']
#         else:
#             Lambda_train = pred_nug._info['nugs_o'][i, :]/info['nu_hat']
#             mult_train = np.concatenate((info['mult'], mult))
#             mc = mean_cand[i, :]
#             Z0 = np.concatenate((info['Z0'], mc), axis=0)

#         C = cov_gen(X1=theta_train, theta=info['theta'])
#         Ki = cholesky(C + np.diag(Lambda_train / mult_train + info['eps']))
#         Ki = np.linalg.inv((Ki))
#         Ki = Ki @ Ki.T

#         kx = cov_gen(X1=theta, X2=theta_train, theta=info['theta'])
#         if info['trendtype'] == 'SK':
#             sd2 = info['nu_hat'] * (1 - np.diag(np.dot(kx, np.dot(Ki, kx.T))))
#         else:
#             sd2 = info['nu_hat'] * (1 - np.diag(np.dot(kx, np.dot(Ki, kx.T))) + (1 - np.sum(Ki, axis=0)[None, :] @ kx.T ).flatten()**2/np.sum(Ki) )
        
#         if any(sd2 < 0):
#             sd2 = np.maximum(0, sd2)
                
#         predinfo['mean_o'][i, :] = (info['beta0'] + np.dot(kx, np.dot(Ki, Z0 - info['beta0'])))[None, :]   
#         predinfo['var_o'][i, :] = sd2[None, :]  
        
        
#         if thetaprime is not None:
#             # fitinfo['X0'].shape[0] x thetaprime.shape[0]
#             kxprime = cov_gen(X1=theta_train, X2=thetaprime, theta=info['theta'])
#             if info['trendtype'] == 'SK':
#                 if theta.shape[0] < thetaprime.shape[0]:
#                     cov = info['nu_hat'] * (cov_gen(X1=theta, X2=thetaprime, theta=info['theta']) - kx @ Ki @ kxprime)
#                 else:
#                     cov = info['nu_hat'] * (cov_gen(X1=theta, X2=thetaprime, theta=info['theta']) - kx @ (Ki @ kxprime))
#             else:
#                 if theta.shape[0] < thetaprime.shape[0]:
#                     cov = info['nu_hat'] * (cov_gen(X1=theta, X2=thetaprime, theta=info['theta']) - kx @ Ki @ kxprime + \
#                                             ((1 - np.sum(Ki, axis=0)[None, :] @ kx.T).T @ (1 - np.sum(Ki, axis=0)[None, :] @ kxprime))/np.sum(Ki))
#                                                 #np.dot(1 - np.sum(Ki, axis=0)[None, :] @ kx.T, 1 - np.sum(Ki, axis=0)[None, :] @ kxprime)/np.sum(Ki))
#                 else:
#                     cov = info['nu_hat'] * (cov_gen(X1=theta, X2=thetaprime, theta=info['theta']) - kx @ (Ki @ kxprime) + \
#                                                 np.dot((1 - np.sum(Ki, axis=0)[None, :] @ kx.T).T, 1 - np.sum(Ki, axis=0)[None, :] @ kxprime)/np.sum(Ki))
        
#             predinfo['cov_o'][i, :, :] = cov
    
#     predinfo['S'] = np.full((d, d, n), np.nan)
#     predinfo['mean'] = fitinfo['h'] + fitinfo['G']@(fitinfo['B']@predinfo['mean_o'])
#     for i in range(0, theta.shape[0]):
#         C = np.diag(predinfo['var_o'][:, i])
#         predinfo['S'][:, :, i] = fitinfo['G']@fitinfo['B']@C@fitinfo['B'].T@fitinfo['G']

#     return