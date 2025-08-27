"""Contains supplemental methods for acquisitionn funcs."""

import numpy as np
from PUQ.surrogate import emulator
from numpy.linalg import inv, det

def impute(ct, x, fE, tE, reps, emu, rnd_str):
        
    # NEW PART
    if fE.shape[0] > 1:
        cpred = emu.predict(x=x, theta=ct)
        cm = cpred.mean()
        cS = cpred._info['S']
        cR = cpred._info['R']
        fnoise = rnd_str.multivariate_normal(mean=cm.flatten(), 
                                               cov=cS[:, :, 0] + cR[:, :, 0], 
                                               size=reps)

        tE = np.concatenate([tE, np.repeat(ct, reps, axis=0)])
        fE = np.concatenate([fE, fnoise.T], axis=1)
    else:
        cpred = emu.predict(x=x, theta=ct)
        cm = cpred.mean()
        cv = cpred._info['var_noisy']
        fnoise = rnd_str.normal(loc=cm.flatten(), 
                                  scale=np.sqrt(cv.flatten()), 
                                  size=reps)

        tE = np.concatenate([tE, np.repeat(ct, reps, axis=0)])
        fE = np.concatenate([fE, fnoise.reshape(1, reps)], axis=1)
    
    return fE, tE

def impute_CL(ct, x, fE, tE, reps, liar):

    #print(fE)
    d = fE.shape[0]
    
    tE = np.concatenate([tE, np.repeat(ct, reps, axis=0)])
    fE = np.concatenate([fE, np.repeat(liar, reps).reshape(d, reps)], axis=1)
    
    #print(fE)
    
    return fE, tE

def build_emulator(x, theta, f, pcset):
    
    print(theta.shape)
    emu = emulator(x=x, 
                   theta=theta, 
                   f=f,                
                   method="pcHetGP",
                   args={'lower':None, 'upper':None,
                          'noiseControl':{'k_theta_g_bounds': (1, 100), 'g_max': 1e2, 'g_bounds': (1e-6, 1)}, 
                          'init':{}, 
                          'known':{}, 
                           'settings':{"linkThetas": 'joint', "logN": True, "initStrategy": 'residuals', 
                                     "checkHom": True, "penalty": True, "trace": 0, "return.matrices": True, 
                                     "return.hom": False, "factr": 1e9},
                           'pc_settings':pcset})
    return emu

def compute_ivar(emu, ttest, x, obs, obsvar):
    
    testP = emu.predict(x=x, theta=ttest)
    d = len(x)
    obsvar3d = obsvar.reshape(1, d, d) 
    
    # ntest x d
    mu = testP._info['mean'].T 
    S =  testP._info['S']
    St = np.transpose(S, (2, 0, 1))

    # ntest x d x d
    M = St + 0.5*obsvar3d
    N = St + obsvar3d

    f = multiple_pdfs(obs, mu, M)
    g = multiple_pdfs(obs, mu, N)

    coef = (1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det(obsvar))))
    
    # ivar compute
    ivar = np.sum(coef*f - g**2)
    return ivar


def multiple_pdfs(x, means, covs):
    # Cite: http://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/

    # NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs = 1.0 / vals

    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us = vecs * np.sqrt(valsinvs)[:, None]
    devs = x - means

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs = np.einsum("ni,nij->nj", devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas = np.sum(np.square(devUs), axis=1)

    # Compute and broadcast scalar normalizers.
    dim = len(vals[0])
    log2pi = np.log(2 * np.pi)
    return np.exp(-0.5 * (dim * log2pi + mahas + logdets))


def multiple_determinants(covs):
    vals, vecs = np.linalg.eigh(covs)
    # Compute the log determinants across the second axis.
    # dets = np.prod(vals, axis=1)
    logdets = np.sum(np.log(vals), axis=1)
    return np.exp(logdets)  # dets#np.exp(logdets)

