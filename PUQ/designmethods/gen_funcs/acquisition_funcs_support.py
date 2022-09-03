"""Contains supplemental methods for gen function in persistent_surmise_calib.py."""

import numpy as np
import scipy.stats as sps


def importance_sample(emu, x, thetatest, obs, obsvar3d, coef):
    emupredict     = emu.predict(x, thetatest)
    emumean        = emupredict.mean()   
    emuvar, is_cov = get_emuvar(emupredict)
    emumeanT       = emumean.T
    emuvarT        = emuvar.transpose(1, 0, 2)
    var_obsvar1    = emuvarT + obsvar3d 
    var_obsvar2    = emuvarT + 0.5*obsvar3d 
    postvar        = compute_postvar(obs, emumeanT, var_obsvar1, var_obsvar2, coef)
    weights        = postvar/np.sum(postvar)
    postmean       = multiple_pdfs(obs, emumeanT, var_obsvar1)
    
    # ids            = np.random.choice(np.arange(0, len(postvar)), p=weights, size=5000, replace=False)   
    # weightssum     = (1/postvar)/np.sum(1/postvar)
    weights        = postmean/np.sum(postmean)
    ids            = np.random.choice(np.arange(0, len(postmean)), p=weights, size=5000, replace=False)   
    # weightssum     = (1/postmean)/np.sum(1/postmean)
    
    #ids = np.random.choice(np.arange(0, len(thetatest)), size=5000, replace=False)   
    weightssum = np.ones(len(thetatest))
    return weightssum[ids], thetatest[ids, :]
    
def compute_likelihood(emumean, emuvar, obs, obsvar, is_cov):

    if emumean.shape[0] == 1:
        
        emuvar = emuvar.reshape(emumean.shape)
        ll = sps.norm.pdf(obs-emumean, 0, np.sqrt(obsvar + emuvar))
    else:
        ll = np.zeros(emumean.shape[1])
        for i in range(emumean.shape[1]):
            mean = emumean[:, i] #[emumean[0, i], emumean[1, i]]
    
            if is_cov:
                cov = emuvar[:, i, :] + obsvar
            else:
                cov = np.diag(emuvar[:, i]) + obsvar 

            rnd = sps.multivariate_normal(mean=mean, cov=cov)
            ll[i] = rnd.pdf(obs) #rnd.pdf([obs[0, 0], obs[0, 1]])

    return ll

def compute_postvar(obs, emumean, covmat1, covmat2, coef):
    # n_x = emumean.shape[0]
    # if n_x > 1:
    #     diags = np.diag(obsvar)
    # else:
    #     diags = 1*obsvar
    # coef = (2**n_x)*(np.sqrt(np.pi)**n_x)*np.sqrt(np.prod(diags))

    part1 = multiple_pdfs(obs, emumean, covmat2)
    part2 = multiple_pdfs(obs, emumean, covmat1)
    
    # part1 = compute_likelihood(emumean, emuvar, obs, 0.5*obsvar, is_cov)
    part1 = part1*(1/coef)

    # part2 = compute_likelihood(emumean, emuvar, obs, obsvar, is_cov)
    part2 = part2**2

    return part1 - part2

def compute_eivar_fig(obsvar, summatrix2, 
                  summatrix, emuphi, emumean, emuvar, obs, is_cov):
    
    rndpdf2  = multiple_pdfs(obs, emumean, summatrix2)
    denum2   = obsvar
    # See Eq. 31 
    covmat1 = (summatrix + emuphi)*0.5
    covmat2 = summatrix - emuphi
    
    rndpdf  = multiple_pdfs(obs, emumean, covmat1)

    
    denum   = multiple_determinants(covmat2)
    part2   = rndpdf/np.sqrt(denum)
    # print(part2.shape)
    return (np.sum(rndpdf2/np.sqrt(denum2)) - np.sum(part2))*(1/(2*np.pi**(0.5)))


def compute_eivar(summatrix, emuphi, emumean, emuvar, obs, is_cov):

    # See Eq. 31 
    covmat1 = (summatrix + emuphi)*0.5
    covmat2 = summatrix - emuphi
    
    rndpdf  = multiple_pdfs(obs, emumean, covmat1)

    denum   = multiple_determinants(covmat2)
    part2   = rndpdf/np.sqrt(denum)
    # print(part2.shape)
    return - np.sum(part2)
    
def multiple_pdfs(x, means, covs):
    
    # Cite: http://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
    
    # NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets    = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs   = 1./vals
    
    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us         = vecs * np.sqrt(valsinvs)[:, None]
    devs       = x - means

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs      = np.einsum('ni,nij->nj', devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas      = np.sum(np.square(devUs), axis=1)
    
    # Compute and broadcast scalar normalizers.
    dim        = len(vals[0])
    log2pi     = np.log(2 * np.pi)
    return np.exp(-0.5 * (dim * log2pi + mahas + logdets))

def multiple_determinants(covs):

    vals, vecs = np.linalg.eigh(covs)
    # Compute the log determinants across the second axis.
    #dets = np.prod(vals, axis=1)
    logdets    = np.sum(np.log(vals), axis=1)
    return np.exp(logdets)#dets#np.exp(logdets)
    


def get_emuvar(emupredict):
    is_cov = False
    try:
        emuvar = emupredict.covx() 
        is_cov = True
    except Exception:
        emuvar = emupredict.var()
    
    return emuvar, is_cov

# def compute_eivar_p1(obsvar, emumean, emuvar, obs, is_cov):
#     n_x = emumean.shape[0]
#     if n_x > 1:
#         diags = np.diag(obsvar)
#     else:
#         diags = 1*obsvar
#     denum = (2**n_x)*(np.sqrt(np.pi)**n_x)*np.sqrt(np.prod(diags))

#     part1 = compute_likelihood(emumean, emuvar, obs, 0.5*obsvar, is_cov)
#     part1 = part1*(1/denum)
    
#     return part1

# def compute_eivar(summatrix, emuphi, emumean, emuvar, obs, is_cov):

#     n = emumean.shape[0]
#     ### PART 2 of Eq. 31 ###
#     part2 = np.zeros(n)
#     covmat1 =  (summatrix + emuphi)*0.5
#     covmat2 =  summatrix - emuphi

#     for i in range(n):
#         rndpdf = sps.multivariate_normal.pdf(obs, mean=emumean[i, :], cov=covmat1[i, :, :])
#         #denum2 = coef*np.sqrt(np.linalg.det(covmat2[:, :, i]))
#         denum2 = np.sqrt(np.linalg.det(covmat2[i, :, :]))
#         part2[i] = rndpdf/denum2

#     return - np.sum(part2)