"""PCGP (Higdon et al., 2008). """

import numpy as np
import scipy.optimize as spo
import scipy.linalg as spla
import copy
from PUQ.surrogatesupport.matern_covmat import covmat as __covmat
import torch
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import multiple_pdfs, multiple_determinants

def fit(fitinfo, x, theta, f, epsilonPC=0.001,
        lognugmean=-10, lognugLB=-20, varconstant=None, dampalpha=0.3, eta=10,
        standardpcinfo=None, verbose=0, **kwargs):
    '''
    The purpose of fit is to take information and plug all of our fit
    information into fitinfo, which is a python dictionary.

    Parameters
    ----------
    fitinfo : dict
        A dictionary including the emulation fitting information once
        complete.
        The dictionary is passed by reference, so it returns None.
    x : numpy.ndarray
        An array of inputs. Each row should correspond to a row in f.
    theta : numpy.ndarray
        An array of parameters. Each row should correspond to a column in f.
    f : numpy.ndarray
        An array of responses. Each column in f should correspond to a row in
        theta. Each row in f should correspond to a row in x.
    epsilonPC : scalar
        A parameter to control the number of PCs used.  The suggested range for
        epsilonPC is (0.001, 0.1).  The larger epsilonPC is, the fewer PCs will be
        used.  Note that epsilonPC here is *not* the unexplained variance in
        typical principal component analysis.
    lognugmean : scalar
        A parameter to control the log of the nugget used in fitting the GPs.
        The suggested range for lognugmean is (-12, -4).  The nugget is estimated,
        and this parameter is used to guide the estimation.
    lognugLB : scalar
        A parameter to control the lower bound of the log of the nugget. The
        suggested range for lognugLB is (-24, -12).
    varconstant : scalar
        A multiplying constant to control the inflation (deflation) of additional
        variances if missing values are present. Default is None, the parameter will
        be optimized in such case. A general working range is (np.exp(-4), np.exp(4)).
    dampalpha : scalar
        A parameter to control the rate of increase of variance as amount of missing
        values increases.  Default is 0.3, otherwise an appropriate range is (0, 0.5).
        Values larger than 0.5 are permitted but it leads to poor empirical performance.
    eta : scalar
        A parameter as an upper bound for the additional variance term.  Default is 10.
    standardpcinfo : dict
        A dictionary user supplies that contains information for standardization of `f`,
        in the following format, such that fs = (f - offset) / scale, U are the
        orthogonal basis vectors, and S are the singular values from SVD of `fs`.
        The entry extravar contains the average squared residual for each column (x).
            {'offset': offset,
             'scale': scale,
             'fs': fs,
             'extravar': extravar,
             'U': U,  # optional
             'S': S  # optional
             }

    verbose : scalar
        A parameter to suppress in-method console output.  Use 0 to suppress output,
        use 1 to show output.


    kwargs : dict, optional
        A dictionary containing options. The default is None.

    Returns
    -------
    None.

    '''
    #print('Fitting...')
    f = f.T
    fitinfo['epsilonPC'] = epsilonPC
    hyp1 = lognugmean
    hyp2 = lognugLB
    hypvarconst = np.log(varconstant) if varconstant is not None else None

    fitinfo['dampalpha'] = dampalpha
    fitinfo['eta'] = eta

    fitinfo['theta'] = theta
    fitinfo['f'] = f
    fitinfo['x'] = x

    # Standardize the function evaluations f
    if standardpcinfo is None:
        __standardizef(fitinfo)
    else:
        fitinfo['standardpcinfo'] = standardpcinfo

    # Construct principal components
    __PCs(fitinfo)
    numpcs = fitinfo['pc'].shape[1]

    if verbose > 0:
        print(fitinfo['method'], 'considering ', numpcs, 'PCs')

    # Fit emulators for all PCs
    emulist = __fitGPs(fitinfo, theta, numpcs, hyp1, hyp2, hypvarconst)
    fitinfo['varc_status'] = 'fixed' if varconstant is not None else 'optimized'
    fitinfo['logvarc'] = np.array([emulist[i]['hypvarconst'] for i in range(numpcs)])
    fitinfo['pcstdvar'] = np.exp(fitinfo['logvarc']) * fitinfo['unscaled_pcstdvar']
    fitinfo['emulist'] = emulist

    return

def update(fitinfo, x, theta, f, **kwargs):
    print('Updating...')
    f = f.T
    #print(fitinfo['f'].shape)
    #print(f.shape)
    #print(fitinfo['pc'].shape)

    fitinfo['theta'] = theta
    fitinfo['f'] = f
    fitinfo['x'] = x
    
    standardpcinfo = fitinfo['standardpcinfo']    
    offset = standardpcinfo['offset']
    scale = standardpcinfo['scale']
    fs = (f - offset) / scale
    standardpcinfo['fs'] = fs
    fitinfo['pc'] = fs @ fitinfo['pct']
    emulist = fitinfo['emulist']
    numpcs = fitinfo['pc'].shape[1]
    
    for pcanum in range(0, numpcs):
        subinfo = emulist[pcanum]
        R = __covmat(theta, theta, subinfo['hypcov'])
        subinfo['R'] = (1 - subinfo['nug']) * R + subinfo['nug'] * np.eye(R.shape[0])
        W, V = np.linalg.eigh(subinfo['R'])
        Vh = V / np.sqrt(np.abs(W))
        #sig2ofconst = subinfo['sig2ofconst']
        g = fitinfo['pc'][:, pcanum]
        #fcenter = Vh.T @ g
        subinfo['Vh'] = Vh
        #n = subinfo['R'].shape[0]
        #subinfo['sig2'] = (np.mean(fcenter ** 2) * n + sig2ofconst) / (n + sig2ofconst)
        subinfo['Rinv'] = V @ np.diag(1 / W) @ V.T
        subinfo['pw'] = subinfo['Rinv'] @ g
 
    
def predict(predinfo, fitinfo, x, theta, **kwargs):
    r"""
    Finds prediction at theta and x given the dictionary fitinfo.
    This [emulationpredictdocstring] automatically filled by docinfo.py when
    running updatedocs.py

    Parameters
    ----------
    predinfo : dict
        An arbitary dictionary where you should place all of your prediction
        information once complete. This dictionary is pass by reference, so
        there is no reason to return anything. Keep only stuff that will be
        used by predict. Key elements are

            - `predinfo['mean']` : `predinfo['mean'][k]` is mean of the prediction
              at all x at `theta[k]`.
            - `predinfo['var']` : `predinfo['var'][k]` is variance of the
              prediction at all x at `theta[k]`.
            - `predinfo['cov']` : `predinfo['cov'][k]` is covariance matrix of the prediction
              at all x at `theta[k]`.
            - `predinfo['covhalf']` : if `A = predinfo['covhalf'][k]` then
              `A.T @ A = predinfo['cov'][k]`.

    fitinfo : dict
        An arbitary dictionary where you placed all your important fitting
        information from the fit function above.

    x : array of objects
        An matrix (vector) of inputs for prediction.

    theta :  array of objects
        An matrix (vector) of parameters to prediction.

    kwargs : dict
        A dictionary containing additional options
    """
    return_grad = False
    if (kwargs is not None) and ('return_grad' in kwargs.keys()) and \
            (kwargs['return_grad'] is True):
        return_grad = True
    return_covx = True
    if (kwargs is not None) and ('return_covx' in kwargs.keys()) and \
            (kwargs['return_covx'] is False):
        return_covx = False
    infos = fitinfo['emulist']
    predvecs = np.zeros((theta.shape[0], len(infos)))
    predvars = np.zeros((theta.shape[0], len(infos)))
    predcovs = np.zeros((theta.shape[0], theta.shape[0], len(infos)))
    
    if predvecs.ndim < 1.5:
        predvecs = predvecs.reshape((1, -1))
        predvars = predvars.reshape((1, -1))
    try:
        if x is None or np.all(np.equal(x, fitinfo['x'])) or \
                np.allclose(x, fitinfo['x']):
            xind = np.arange(0, x.shape[0])
            xnewind = np.arange(0, x.shape[0])
        else:
            raise
    except Exception:
        matchingmatrix = np.ones((x.shape[0], fitinfo['x'].shape[0]))
        for k in range(0, x[0].shape[0]):
            try:
                matchingmatrix *= np.isclose(x[:, k][:, None],
                                             fitinfo['x'][:, k])
            except Exception:
                matchingmatrix *= np.equal(x[:, k][:, None],
                                           fitinfo['x'][:, k])
        xind = np.argwhere(matchingmatrix > 0.5)[:, 1]
        xnewind = np.argwhere(matchingmatrix > 0.5)[:, 0]

    rsave = np.array(np.ones(len(infos)), dtype=object)
    rsave_3 = np.array(np.ones(len(infos)), dtype=object)
    # loop over principal components
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            # covariance matrix between new theta and thetas from fit.
            rsave[k] = __covmat(theta,
                                fitinfo['theta'],
                                infos[k]['hypcov'])
            
            rsave_3[k] = __covmat(theta,
                                  theta,
                                  infos[k]['hypcov'])
        # adjusted covariance matrix
        r = (1 - infos[k]['nug']) * np.squeeze(rsave[infos[k]['hypind']])
        r3 = (1 - infos[k]['nug']) * np.squeeze(rsave_3[infos[k]['hypind']])
        try:
            rVh = r @ infos[k]['Vh']
            rVh2 = rVh @ (infos[k]['Vh']).T
        except Exception:
            for i in range(0, len(infos)):
                print((i, infos[i]['hypind']))
            raise ValueError('Something went wrong with fitted components')

        if rVh.ndim < 1.5:
            rVh = rVh.reshape((1, -1))
        if rVh2.ndim < 1.5:
            rVh2 = np.reshape(rVh2, (1, -1))
        predvecs[:, k] = r @ infos[k]['pw']
        predvars[:, k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh ** 2, 1))
        predcovs[:, :, k] = infos[k]['sig2'] * (r3 - rVh @ rVh.T) #np.abs(1 - np.sum(rVh ** 2, 1))
   
    

    # calculate predictive mean and variance
    predinfo['mean'] = np.full((x.shape[0], theta.shape[0]), np.nan)
    predinfo['var'] = np.full((x.shape[0], theta.shape[0]), np.nan)
    predinfo['covx'] = np.full((theta.shape[0], theta.shape[0]), np.nan)
    pctscale = (fitinfo['pcti'].T * fitinfo['standardpcinfo']['scale']).T
    # pctscale = (fitinfo['pct'].T * fitinfo['standardpcinfo']['scale']).T
    predinfo['mean'][xnewind, :] = ((predvecs @ pctscale[xind, :].T) +
                                    fitinfo['standardpcinfo']['offset'][xind]).T
    predinfo['var'][xnewind, :] = ((fitinfo['standardpcinfo']['extravar'][xind] +
                                    predvars @ (pctscale[xind, :] ** 2).T)).T
    
    predinfo['covx'] = predcovs[:, :, 0] * (pctscale[:, :] ** 2)
    
    #print(predinfo['var'])
    #print(predinfo['cov'])

    predinfo['extravar'] = 1 * fitinfo['standardpcinfo']['extravar'][xind]
    predinfo['predvars'] = 1 * predvars
    predinfo['predvecs'] = 1 * predvecs
    predinfo['phi'] = 1 * pctscale[xind, :]


    return



def predictlpdf(predinfo, f, return_grad=False, addvar=0, **kwargs):
    totvar = addvar + predinfo['extravar']
    rf = ((f.T - predinfo['mean'].T) * (1 / np.sqrt(totvar))).T
    Gf = predinfo['phi'].T * (1 / np.sqrt(totvar))
    Gfrf = Gf @ rf
    Gf2 = Gf @ Gf.T
    likv = np.sum(rf ** 2, 0)
    if return_grad:
        rf2 = -(predinfo['mean_gradtheta'].transpose(2, 1, 0) *
                (1 / np.sqrt(totvar))).transpose(2, 1, 0)
        Gfrf2 = (Gf @ rf2.transpose(1, 0, 2)).transpose(1, 0, 2)
        dlikv = 2 * np.sum(rf2.transpose(2, 1, 0) * rf.transpose(1, 0), 2).T
    for c in range(0, predinfo['predvars'].shape[0]):
        w, v = np.linalg.eig(np.diag(1 / (predinfo['predvars'][c, :])) + Gf2)
        term1 = (v * (1 / w)) @ (v.T @ Gfrf[:, c])

        likv[c] -= Gfrf[:, c].T @ term1
        likv[c] += np.sum(np.log(predinfo['predvars'][c, :]))
        likv[c] += np.sum(np.log(w))
        if return_grad:
            Si = (v * (1 / w)) @ v.T
            grt = (predinfo['predvars_gradtheta'][c, :, :].T / predinfo['predvars'][c, :]).T
            dlikv[c, :] += np.sum(grt, 0)
            grt = (-grt.T / predinfo['predvars'][c, :]).T
            dlikv[c, :] += np.diag(Si) @ grt
            term2 = (term1 / predinfo['predvars'][c, :]) ** 2
            dlikv[c, :] -= 2 * Gfrf2[:, c, :].T @ term1
            dlikv[c, :] -= term2 @ (predinfo['predvars_gradtheta'][c, :, :])
    if return_grad:
        return (-likv / 2).reshape(-1, 1), (-dlikv / 2)
    else:
        return (-likv / 2).reshape(-1, 1)





def __standardizef(fitinfo, offset=None, scale=None):
    r'''Standardizes f by creating offset, scale and fs.'''
    # Extracting from input dictionary
    f = fitinfo['f']
    epsilonPC = fitinfo['epsilonPC']

    if (offset is not None) and (scale is not None):
        if offset.shape[0] == f.shape[1] and scale.shape[0] == f.shape[1]:
            if np.any(np.nanmean(np.abs(f - offset) / scale, 1) > 4):
                offset = None
                scale = None
        else:
            offset = None
            scale = None
    if offset is None or scale is None:
        offset = np.zeros(f.shape[1])
        scale = np.zeros(f.shape[1])
        for k in range(0, f.shape[1]):
            offset[k] = np.nanmean(f[:, k])
            scale[k] = np.nanstd(f[:, k]) / np.sqrt(1-np.isnan(f[:, k]).mean())
            if scale[k] == 0:
                print(f)
                raise ValueError("You have a row that is non-varying.")

    fs = np.zeros(f.shape)
    fs = (f - offset) / scale

    # Assigning new values to the dictionary
    U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
    Sp = S ** 2 - epsilonPC
    Up = U[:, Sp > 0]

    extravar = np.nanmean((fs - fs @ Up @ Up.T) ** 2, 0) * (scale ** 2)

    standardpcinfo = {'offset': offset,
                      'scale': scale,
                      'fs': fs,
                      'U': U,
                      'S': S,
                      'extravar': extravar
                      }

    fitinfo['standardpcinfo'] = standardpcinfo
    return


def __PCs(fitinfo):
    "Apply PCA to reduce the dimension of `f`."
    # Extracting from input dictionary
    f = fitinfo['f']
    epsilonPC = fitinfo['epsilonPC']
    fs = fitinfo['standardpcinfo']['fs']
    
    if 'U' in fitinfo['standardpcinfo']:
        U = fitinfo['standardpcinfo']['U']
        S = fitinfo['standardpcinfo']['S']
    else:
        U, S, _ = np.linalg.svd(fs.T, full_matrices=False)
    Sp = S ** 2 - epsilonPC
    pct = U[:, Sp > 0]
    pcw = np.sqrt(Sp[Sp > 0])

    pcstdvar = np.zeros((f.shape[0], pct.shape[1]))
    fitinfo['pcw'] = pcw
    fitinfo['pcto'] = 1 * pct
    effn = np.sum(np.clip(1 - pcstdvar, 0, 1))
    fitinfo['pct'] = pct * pcw / np.sqrt(effn)
    fitinfo['pcti'] = pct * (np.sqrt(effn) / pcw)
    # fitinfo['pc'] = pc * (np.sqrt(effn) / pcw)
    fitinfo['pc'] = fs @ fitinfo['pct']
    fitinfo['unscaled_pcstdvar'] = pcstdvar
    return


def __fitGPs(fitinfo, theta, numpcs, hyp1, hyp2, varconstant):
    """Fit emulators for all principle components."""
    if 'emulist' in fitinfo.keys():
        hypstarts = np.zeros((numpcs, fitinfo['emulist'][0]['hyp'].shape[0]))
        hypinds = -1 * np.ones(numpcs)
        for pcanum in range(0, min(numpcs, len(fitinfo['emulist']))):
            hypstarts[pcanum, :] = fitinfo['emulist'][pcanum]['hyp']
            hypinds[pcanum] = fitinfo['emulist'][pcanum]['hypind']
    else:
        hypstarts = None
        hypinds = -1 * np.ones(numpcs)

    emulist = [dict() for x in range(0, numpcs)]
    for iters in range(0, 3):
        for pcanum in range(0, numpcs):
            if np.sum(hypinds == np.array(range(0, numpcs))) > 0.5:
                hypwhere = np.where(hypinds == np.array(range(0, numpcs)))[0]
                emulist[pcanum] = __fitGP1d(theta=theta,
                                            g=fitinfo['pc'][:, pcanum],
                                            hyp1=hyp1,
                                            hyp2=hyp2,
                                            hypvarconst=varconstant,
                                            gvar=fitinfo['unscaled_pcstdvar'][:, pcanum],
                                            dampalpha=fitinfo['dampalpha'],
                                            eta=fitinfo['eta'],
                                            hypstarts=hypstarts[hypwhere, :],
                                            hypinds=hypwhere,
                                            sig2ofconst=0.01)
            else:
                emulist[pcanum] = __fitGP1d(theta=theta,
                                            g=fitinfo['pc'][:, pcanum],
                                            hyp1=hyp1,
                                            hyp2=hyp2,
                                            hypvarconst=varconstant,
                                            gvar=fitinfo['unscaled_pcstdvar'][:, pcanum],
                                            dampalpha=fitinfo['dampalpha'],
                                            eta=fitinfo['eta'],
                                            sig2ofconst=0.01)
                hypstarts = np.zeros((numpcs, emulist[pcanum]['hyp'].shape[0]))
            emulist[pcanum]['hypind'] = min(pcanum, emulist[pcanum]['hypind'])
            hypstarts[pcanum, :] = emulist[pcanum]['hyp']
            if emulist[pcanum]['hypind'] < -0.5:
                emulist[pcanum]['hypind'] = 1 * pcanum
            hypinds[pcanum] = 1 * emulist[pcanum]['hypind']
    return emulist


def __fitGP1d(theta, g, hyp1, hyp2, hypvarconst, gvar=None, dampalpha=None, eta=None,
              hypstarts=None, hypinds=None, sig2ofconst=None):
    """Return a fitted model from the emulator model using smart method."""
    hypvarconstmean = 4 if hypvarconst is None else hypvarconst
    hypvarconstLB = -8 if hypvarconst is None else hypvarconst - 0.5
    hypvarconstUB = 8 if hypvarconst is None else hypvarconst + 0.5

    subinfo = {}
    subinfo['hypregmean'] = np.append(0 + 0.5 * np.log(theta.shape[1]) +
                                      np.log(np.std(theta, 0)), (0, hypvarconstmean, hyp1))
    subinfo['hypregLB'] = np.append(-4 + 0.5 * np.log(theta.shape[1]) +
                                    np.log(np.std(theta, 0)), (-12, hypvarconstLB, hyp2))

    subinfo['hypregUB'] = np.append(4 + 0.5 * np.log(theta.shape[1]) +
                                    np.log(np.std(theta, 0)), (2, hypvarconstUB, -8))
    subinfo['hypregstd'] = (subinfo['hypregUB'] - subinfo['hypregLB']) / 8
    subinfo['hypregstd'][-3] = 2
    subinfo['hypregstd'][-1] = 4
    subinfo['hyp'] = 1 * subinfo['hypregmean']
    nhyptrain = np.max(np.min((20 * theta.shape[1], theta.shape[0])))
    if theta.shape[0] > nhyptrain:
        thetac = np.random.choice(theta.shape[0], nhyptrain, replace=False)
    else:
        thetac = range(0, theta.shape[0])
    subinfo['theta'] = theta[thetac, :]
    subinfo['g'] = g[thetac]

    # maxgvar = np.max(gvar)
    # gvar = gvar / ((np.abs(maxgvar*1.001 - gvar)) ** dampalpha)

    gvar = np.minimum(eta, gvar / ((1 - gvar)**dampalpha))

    #print(gvar)
    subinfo['sig2ofconst'] = sig2ofconst
    subinfo['gvar'] = gvar[thetac]
    hypind0 = -1

    L0 = __negloglik(subinfo['hyp'], subinfo)
    if hypstarts is not None:
        L0 = __negloglik(subinfo['hyp'], subinfo)
        for k in range(0, hypstarts.shape[0]):
            L1 = __negloglik(hypstarts[k, :], subinfo)
            if L1 < L0:
                subinfo['hyp'] = hypstarts[k, :]
                L0 = 1 * L1
                hypind0 = hypinds[k]

    if hypind0 > -0.5 and hypstarts.ndim > 1:
        dL = __negloglikgrad(subinfo['hyp'], subinfo)
        scalL = np.std(hypstarts, 0) * hypstarts.shape[0] / \
            (1 + hypstarts.shape[0]) + (1 / (1 + hypstarts.shape[0]) * subinfo['hypregstd'])
        if np.sum((dL * scalL) ** 2) < 1.25 * \
                (subinfo['hyp'].shape[0] + 5 * np.sqrt(subinfo['hyp'].shape[0])):
            skipop = True
        else:
            skipop = False
    else:
        skipop = False

    if not skipop:
        def scaledlik(hypv):
            hyprs = subinfo['hypregmean'] + hypv * subinfo['hypregstd']
            return __negloglik(hyprs, subinfo)

        def scaledlikgrad(hypv):
            hyprs = subinfo['hypregmean'] + hypv * subinfo['hypregstd']
            return __negloglikgrad(hyprs, subinfo) * subinfo['hypregstd']

        newLB = (subinfo['hypregLB'] - subinfo['hypregmean']) / subinfo['hypregstd']
        newUB = (subinfo['hypregUB'] - subinfo['hypregmean']) / subinfo['hypregstd']

        newhyp0 = (subinfo['hyp'] - subinfo['hypregmean']) / subinfo['hypregstd']

        opval = spo.minimize(scaledlik,
                             newhyp0,
                             method='L-BFGS-B',
                             options={'gtol': 0.1},
                             jac=scaledlikgrad,
                             bounds=spo.Bounds(newLB, newUB))

        hypn = subinfo['hypregmean'] + opval.x * subinfo['hypregstd']
        likdiff = (L0 - __negloglik(hypn, subinfo))
    else:
        likdiff = 0
    if hypind0 > -0.5 and (2 * likdiff) < 1.25 * \
            (subinfo['hyp'].shape[0] + 5 * np.sqrt(subinfo['hyp'].shape[0])):
        subinfo['hypcov'] = subinfo['hyp'][:-2]
        subinfo['hypvarconst'] = subinfo['hyp'][-2]
        subinfo['hypind'] = hypind0
        subinfo['nug'] = np.exp(subinfo['hyp'][-1]) / (1 + np.exp(subinfo['hyp'][-1]))

        R = __covmat(theta, theta, subinfo['hypcov'])

        subinfo['R'] = (1 - subinfo['nug']) * R + subinfo['nug'] * np.eye(R.shape[0])
        if gvar is not None:
            subinfo['R'] += np.exp(subinfo['hypvarconst'])*np.diag(gvar)

        W, V = np.linalg.eigh(subinfo['R'])
        Vh = V / np.sqrt(np.abs(W))
        fcenter = Vh.T @ g
        subinfo['Vh'] = Vh
        n = subinfo['R'].shape[0]
        subinfo['sig2'] = (np.mean(fcenter ** 2) * n + sig2ofconst) / (n + sig2ofconst)
        subinfo['Rinv'] = V @ np.diag(1 / W) @ V.T
    else:
        subinfo['hyp'] = hypn
        subinfo['hypind'] = -1
        subinfo['hypcov'] = subinfo['hyp'][:-2]
        subinfo['hypvarconst'] = subinfo['hyp'][-2]
        subinfo['nug'] = np.exp(subinfo['hyp'][-1]) / (1 + np.exp(subinfo['hyp'][-1]))

        R = __covmat(theta, theta, subinfo['hypcov'])
        subinfo['R'] = (1 - subinfo['nug']) * R + subinfo['nug'] * np.eye(R.shape[0])
        if gvar is not None:
            subinfo['R'] += np.exp(subinfo['hypvarconst'])*np.diag(gvar)
        n = subinfo['R'].shape[0]
        W, V = np.linalg.eigh(subinfo['R'])
        Vh = V / np.sqrt(np.abs(W))
        fcenter = Vh.T @ g
        subinfo['sig2'] = (np.mean(fcenter ** 2) * n + sig2ofconst) / (n + sig2ofconst)
        subinfo['Rinv'] = Vh @ Vh.T
        subinfo['Vh'] = Vh
    subinfo['pw'] = subinfo['Rinv'] @ g
    return subinfo


def __negloglik(hyp, info):
    """Return penalized log likelihood of single demensional GP model."""
    R0 = __covmat(info['theta'], info['theta'], hyp[:-2])
    nug = np.exp(hyp[-1]) / (1 + np.exp(hyp[-1]))
    R = (1 - nug) * R0 + nug * np.eye(info['theta'].shape[0])

    if info['gvar'] is not None:
        R += np.exp(hyp[-2])*np.diag(info['gvar'])
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['g']
    n = info['g'].shape[0]

    sig2ofconst = info['sig2ofconst']
    sig2hat = (n * np.mean(fcenter ** 2) + sig2ofconst) / (n + sig2ofconst)
    negloglik = 1 / 2 * np.sum(np.log(np.abs(W))) + 1 / 2 * n * np.log(sig2hat)
    negloglik += 0.5 * np.sum(((10 ** (-8) + hyp - info['hypregmean']) /
                               (info['hypregstd'])) ** 2)
    return negloglik


def __negloglikgrad(hyp, info):
    """Return gradient of the penalized log likelihood of single demensional
    GP model."""
    R0, dR = __covmat(info['theta'], info['theta'], hyp[:-2], True)
    nug = np.exp(hyp[-1]) / (1 + np.exp(hyp[-1]))
    R = (1 - nug) * R0 + nug * np.eye(info['theta'].shape[0])
    dR = (1 - nug) * dR
    dRappend2 = nug / (1 + np.exp(hyp[-1])) * (-R0 + np.eye(info['theta'].shape[0]))

    if info['gvar'] is not None:
        R += np.exp(hyp[-2]) * np.diag(info['gvar'])
        dRappend1 = np.exp(hyp[-2]) * np.diag(info['gvar'])
    else:
        dRappend1 = 0 * np.eye(info['theta'].shape[0])

    dR = np.append(dR, dRappend1[:, :, None], axis=2)
    dR = np.append(dR, dRappend2[:, :, None], axis=2)
    W, V = np.linalg.eigh(R)
    Vh = V / np.sqrt(np.abs(W))
    fcenter = Vh.T @ info['g']
    n = info['g'].shape[0]

    sig2ofconst = info['sig2ofconst']
    sig2hat = (n * np.mean(fcenter ** 2) + sig2ofconst) / (n + sig2ofconst)
    dnegloglik = np.zeros(dR.shape[2])
    Rinv = Vh @ Vh.T

    for k in range(0, dR.shape[2]):
        dsig2hat = - np.sum((Vh @
                             np.multiply.outer(fcenter, fcenter) @
                             Vh.T) * dR[:, :, k]) / (n + sig2ofconst)
        dnegloglik[k] += 0.5 * n * dsig2hat / sig2hat
        dnegloglik[k] += 0.5 * np.sum(Rinv * dR[:, :, k])

    dnegloglik += (10 ** (-8) +
                   hyp - info['hypregmean']) / ((info['hypregstd']) ** 2)
    return dnegloglik


def postphimat(fitinfo, n_x, theta, obs, obsvar, theta_cand, covmat_ref, rVh_1_3D, pred_mean):


    # n_x       = len(x)
    n_tot_ref = theta.shape[0]
    n_ref     = int(n_tot_ref/n_x)
    n_t       = fitinfo['theta'].shape[0]

    infos = fitinfo['emulist']
    predvars_cand = np.zeros((theta_cand.shape[0], len(infos)))

    # n_ref x n_cand
    rsave_2 = np.array(np.ones(len(infos)), dtype=object)
    # n_cand x n_t
    rsave_4 = np.array(np.ones(len(infos)), dtype=object)

    # loop over principal components
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            rsave_2[k] = __covmat(theta,
                                  theta_cand,
                                  infos[k]['hypcov'])

            rsave_4[k] = __covmat(theta_cand,
                                  fitinfo['theta'],
                                  infos[k]['hypcov'])

        # adjusted covariance matrix
        # n_tot_ref 
        r_2 = (1 - infos[k]['nug']) * np.squeeze(rsave_2[infos[k]['hypind']])
        # n_t
        r_4 = (1 - infos[k]['nug']) * np.squeeze(rsave_4[infos[k]['hypind']])

        try:
            rVh_4 = r_4.reshape(1, len(fitinfo['theta'])) @ infos[k]['Vh']
            
        except Exception:
            for i in range(0, len(infos)):
                print((i, infos[i]['hypind']))
            raise ValueError('Something went wrong with fitted components')


        r_2_3D   = r_2.reshape(n_ref, n_x, 1)
        rVh_4_3D = rVh_4.reshape(1, n_t, 1)
        
        # rVh_1_3D: n_ref x n_x x n_t

        # n_ref x n_x x 1
        cov3D = np.matmul(rVh_1_3D, rVh_4_3D)
        
        # n_ref x n_x x 1
        cov_cand_3D = infos[k]['sig2'] * (r_2_3D - cov3D)
        
        predvars_cand[:, k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh_4 ** 2, 1))
        predvars_cand[:, k] += infos[k]['nug']

    
    # calculate candidate variance
    pctscale     = (fitinfo['pcti'].T * fitinfo['standardpcinfo']['scale']).T

    # n_ref x n_x x 1
    cov_cand_3D  = cov_cand_3D*(pctscale[:, :] ** 2)
    # n_ref x 1 x n_x
    cov_cand_3DT = np.transpose(cov_cand_3D, (0, 2, 1)) 
    
    # (1 x 1)
    var_cand = ((fitinfo['standardpcinfo']['extravar'][:] + predvars_cand @ (pctscale[:, :] ** 2).T)).T
    
    # (n_ref x n_x x n_x)
    Phi3D = (cov_cand_3D * cov_cand_3DT)/var_cand
    cov1  = 0.5*(covmat_ref + Phi3D)
    cov2  = covmat_ref - Phi3D
    p1    = multiple_pdfs(obs, pred_mean, cov1)
    det1  = multiple_determinants(cov2)
    eivar = np.sum((1/((2**n_x)*(np.sqrt(np.pi)**n_x)*np.sqrt(det1)))*p1)
    return eivar

def temp_postphimat(fitinfo, n_x, theta, obs, obsvar):

    n_tot_ref = theta.shape[0]
    n_ref     = int(n_tot_ref/n_x)
    n_t       = fitinfo['theta'].shape[0]
    
    infos = fitinfo['emulist']
    predmean_ref = np.zeros((theta.shape[0], len(infos)))
    predvars_ref = np.zeros((theta.shape[0], len(infos)))

    if predmean_ref.ndim < 1.5:
        predmean_ref = predmean_ref.reshape((1, -1))
        predvars_ref = predvars_ref.reshape((1, -1))

    # n_ref x n_t
    rsave_1 = np.array(np.ones(len(infos)), dtype=object)
    # n_ref x n_ref
    rsave_3 = np.array(np.ones(len(infos)), dtype=object)

    # loop over principal components
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            # covariance matrix between new theta and thetas from fit.
            rsave_1[k] = __covmat(theta,
                                fitinfo['theta'],
                                infos[k]['hypcov'])
            
            rsave_3[k] = __covmat(theta,
                                  theta,
                                  infos[k]['hypcov'])


        # adjusted covariance matrix
        r_1 = (1 - infos[k]['nug']) * np.squeeze(rsave_1[infos[k]['hypind']])
        r_3 = (1 - infos[k]['nug']) * np.squeeze(rsave_3[infos[k]['hypind']])

        try:
            rVh_1 = r_1 @ infos[k]['Vh']
   
        except Exception:
            for i in range(0, len(infos)):
                print((i, infos[i]['hypind']))
            raise ValueError('Something went wrong with fitted components')


        if rVh_1.ndim < 1.5:
            rVh_1 = rVh_1.reshape((1, -1))
            

        id_row = np.arange(0, n_tot_ref)
        id_col = np.arange(0, n_tot_ref).reshape(n_ref, n_x)
        id_col = np.repeat(id_col, repeats=n_x, axis=0)
        
        r_3_3D = r_3[id_row[:, None], id_col].reshape(n_ref, n_x, n_x)

        rVh_1_3d = rVh_1.reshape(n_ref, n_x, n_t)
        rVh_1_3dT = np.transpose(rVh_1_3d, (0, 2, 1))
        cov3D = np.matmul(rVh_1_3d, rVh_1_3dT)
        cov_ref_3D = infos[k]['sig2'] * (r_3_3D - cov3D)
        predmean_ref[:, k] = r_1 @ infos[k]['pw']

    
    # calculate predictive mean and variance
    pctscale     = (fitinfo['pcti'].T * fitinfo['standardpcinfo']['scale']).T
    Smat3D       = cov_ref_3D*(pctscale[:, :] ** 2)


    pred_mean = ((predmean_ref @ pctscale.T) + fitinfo['standardpcinfo']['offset']).T
    pred_mean = pred_mean.reshape(n_ref, n_x)

    #obsvar3D  = obsvar.reshape(1, n_x, n_x)
    #print(Smat3D.shape)
    #covmat_ref = Smat3D + obsvar3D
    covmat_ref = Smat3D + obsvar
    return covmat_ref, rVh_1_3d, pred_mean

def postphimat2(fitinfo, x, theta, obs, obsvar, theta_cand):


    n_x       = len(x)
    n_tot_ref = theta.shape[0]
    n_ref     = int(n_tot_ref/n_x)
    n_cand    = theta_cand[0]
    n_t       = fitinfo['theta'].shape[0]
    
    
    predinfo = {}
    infos = fitinfo['emulist']
    predmean_ref = np.zeros((theta.shape[0], len(infos)))
    predvars_ref = np.zeros((theta.shape[0], len(infos)))
    predvars_cand = np.zeros((theta_cand.shape[0], len(infos)))

    if predmean_ref.ndim < 1.5:
        predmean_ref = predmean_ref.reshape((1, -1))
        predvars_ref = predvars_ref.reshape((1, -1))

    # n_ref x n_t
    rsave_1 = np.array(np.ones(len(infos)), dtype=object)
    # n_ref x n_cand
    rsave_2 = np.array(np.ones(len(infos)), dtype=object)
    # n_ref x n_ref
    rsave_3 = np.array(np.ones(len(infos)), dtype=object)
    # n_cand x n_t
    rsave_4 = np.array(np.ones(len(infos)), dtype=object)

    # loop over principal components
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            # covariance matrix between new theta and thetas from fit.
            rsave_1[k] = __covmat(theta,
                                fitinfo['theta'],
                                infos[k]['hypcov'])
            
            rsave_2[k] = __covmat(theta,
                                  theta_cand,
                                  infos[k]['hypcov'])
            rsave_3[k] = __covmat(theta,
                                  theta,
                                  infos[k]['hypcov'])
            rsave_4[k] = __covmat(theta_cand,
                                  fitinfo['theta'],
                                  infos[k]['hypcov'])

        # adjusted covariance matrix
        r_1 = (1 - infos[k]['nug']) * np.squeeze(rsave_1[infos[k]['hypind']])
        r_2 = (1 - infos[k]['nug']) * np.squeeze(rsave_2[infos[k]['hypind']])
        r_3 = (1 - infos[k]['nug']) * np.squeeze(rsave_3[infos[k]['hypind']])
        r_4 = (1 - infos[k]['nug']) * np.squeeze(rsave_4[infos[k]['hypind']])

        try:
            rVh_1 = r_1 @ infos[k]['Vh']
            rVh_4 = r_4.reshape(1, len(fitinfo['theta'])) @ infos[k]['Vh']
            
        except Exception:
            for i in range(0, len(infos)):
                print((i, infos[i]['hypind']))
            raise ValueError('Something went wrong with fitted components')


        if rVh_1.ndim < 1.5:
            rVh_1 = rVh_1.reshape((1, -1))
            

        id_row = np.arange(0, n_tot_ref)
        id_col = np.arange(0, n_tot_ref).reshape(n_ref, n_x)
        id_col = np.repeat(id_col, repeats=n_x, axis=0)
        
        r_3_3D = r_3[id_row[:, None], id_col].reshape(n_ref, n_x, n_x)
        r_2_3D = r_2.reshape(n_ref, n_x, 1)
    
        rVh_1_3d = rVh_1.reshape(n_ref, n_x, n_t)
        rVh_1_3dT = np.transpose(rVh_1_3d, (0, 2, 1))
        cov3D = np.matmul(rVh_1_3d, rVh_1_3dT)
        cov_ref_3D = infos[k]['sig2'] * (r_3_3D - cov3D)

        
        predmean_ref[:, k] = r_1 @ infos[k]['pw']
        
        rVh_4_3d = rVh_4.reshape(1, n_t, 1)
        cov3D = np.matmul(rVh_1_3d, rVh_4_3d)
        cov_cand_3D = infos[k]['sig2'] * (r_2_3D - cov3D)
        
        
        #predvars_ref[:, k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh_1 ** 2, 1))
        predvars_cand[:, k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh_4 ** 2, 1))
        predvars_cand[:, k] += infos[k]['nug']
        #cov_ref = infos[k]['sig2'] * (r_3.reshape((theta.shape[0], theta.shape[0])) - rVh_1 @ rVh_1.T)
        #cov_cand = infos[k]['sig2'] * (r_2.reshape((theta.shape[0], theta_cand.shape[0])) - rVh_1 @ rVh_4.T)
      

    
    
    # calculate predictive mean and variance
    predinfo['mean'] = np.full((x.shape[0], int(theta.shape[0]/x.shape[0])), np.nan)
    predinfo['varcand'] = np.full((theta_cand.shape[0], theta_cand.shape[0]), np.nan)

    pctscale     = (fitinfo['pcti'].T * fitinfo['standardpcinfo']['scale']).T
    Smat3D       = cov_ref_3D*(pctscale[:, :] ** 2)
    cov_cand_3D  = cov_cand_3D*(pctscale[:, :] ** 2)
    cov_cand_3DT = np.transpose(cov_cand_3D, (0, 2, 1)) 
    

    predinfo['mean'] = ((predmean_ref @ pctscale.T) +
                                    fitinfo['standardpcinfo']['offset']).T
    predinfo['mean'] = predinfo['mean'].reshape(n_ref, n_x)
    predinfo['varcand'] = ((fitinfo['standardpcinfo']['extravar'][:] +
                                    predvars_cand @ (pctscale[:, :] ** 2).T)).T
    
    Phi3D        = (cov_cand_3D * cov_cand_3DT)/predinfo['varcand']

    
    d     = x.shape[0]
    eivar = 0
    obsvar3D  = obsvar.reshape(1, n_x, n_x)
    cov1 = 0.5*(Smat3D + obsvar3D + Phi3D)
    cov2 = Smat3D + obsvar3D - Phi3D
    cov3 = 0.5*obsvar3D + Smat3D
    det2 = multiple_determinants(obsvar3D)

    obs_mult = np.repeat(obs, predinfo['mean'].shape[0], axis=0)
    #print(predinfo['mean'].shape)
    p2 = multiple_pdfs(obs_mult, predinfo['mean'], cov3)
    p1 = multiple_pdfs(obs_mult, predinfo['mean'], cov1)
    det1 = multiple_determinants(cov2)
    eivar = np.sum((1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det2)))*p2) - np.sum((1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det1)))*p1)

    return eivar


def postphimat3(fitinfo, theta, obs, obsvar, theta_cand):

   

    n_x       = obs.shape[1] 
    n_tot_ref = theta.shape[0]
    n_ref     = int(n_tot_ref/n_x)
    #n_cand    = theta_cand[0]
    n_t       = fitinfo['theta'].shape[0]
    
    
    predinfo = {}
    infos = fitinfo['emulist']
    predmean_ref = np.zeros((theta.shape[0], len(infos)))
    predvars_ref = np.zeros((theta.shape[0], len(infos)))
    predvars_cand = np.zeros((theta_cand.shape[0], len(infos)))

    if predmean_ref.ndim < 1.5:
        predmean_ref = predmean_ref.reshape((1, -1))
        predvars_ref = predvars_ref.reshape((1, -1))

    # n_ref x n_t
    rsave_1 = np.array(np.ones(len(infos)), dtype=object)
    # n_ref x n_cand
    rsave_2 = np.array(np.ones(len(infos)), dtype=object)
    # n_ref x n_ref
    rsave_3 = np.array(np.ones(len(infos)), dtype=object)
    # n_cand x n_t
    rsave_4 = np.array(np.ones(len(infos)), dtype=object)

    # loop over principal components
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            # covariance matrix between new theta and thetas from fit.
            rsave_1[k] = __covmat(theta,
                                fitinfo['theta'],
                                infos[k]['hypcov'])
            
            rsave_2[k] = __covmat(theta,
                                  theta_cand,
                                  infos[k]['hypcov'])
            rsave_3[k] = __covmat(theta,
                                  theta,
                                  infos[k]['hypcov'])
            rsave_4[k] = __covmat(theta_cand,
                                  fitinfo['theta'],
                                  infos[k]['hypcov'])

        # adjusted covariance matrix
        r_1 = (1 - infos[k]['nug']) * np.squeeze(rsave_1[infos[k]['hypind']])
        r_2 = (1 - infos[k]['nug']) * np.squeeze(rsave_2[infos[k]['hypind']])
        r_3 = (1 - infos[k]['nug']) * np.squeeze(rsave_3[infos[k]['hypind']])
        r_4 = (1 - infos[k]['nug']) * np.squeeze(rsave_4[infos[k]['hypind']])

        try:
            rVh_1 = r_1 @ infos[k]['Vh']
            rVh_4 = r_4.reshape(1, len(fitinfo['theta'])) @ infos[k]['Vh']
            
        except Exception:
            for i in range(0, len(infos)):
                print((i, infos[i]['hypind']))
            raise ValueError('Something went wrong with fitted components')


        if rVh_1.ndim < 1.5:
            rVh_1 = rVh_1.reshape((1, -1))
            

        id_row = np.arange(0, n_tot_ref)
        id_col = np.arange(0, n_tot_ref).reshape(n_ref, n_x)
        id_col = np.repeat(id_col, repeats=n_x, axis=0)
        
        r_3_3D = r_3[id_row[:, None], id_col].reshape(n_ref, n_x, n_x)
        r_2_3D = r_2.reshape(n_ref, n_x, 1)
    
        rVh_1_3d = rVh_1.reshape(n_ref, n_x, n_t)
        rVh_1_3dT = np.transpose(rVh_1_3d, (0, 2, 1))
        cov3D = np.matmul(rVh_1_3d, rVh_1_3dT)
        cov_ref_3D = infos[k]['sig2'] * (r_3_3D - cov3D)

        
        predmean_ref[:, k] = r_1 @ infos[k]['pw']
        
        rVh_4_3d = rVh_4.reshape(1, n_t, 1)
        cov3D = np.matmul(rVh_1_3d, rVh_4_3d)
        cov_cand_3D = infos[k]['sig2'] * (r_2_3D - cov3D)
        
        
        #predvars_ref[:, k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh_1 ** 2, 1))
        predvars_cand[:, k] = infos[k]['sig2'] * np.abs(1 - np.sum(rVh_4 ** 2, 1))
        predvars_cand[:, k] += infos[k]['nug']
        #cov_ref = infos[k]['sig2'] * (r_3.reshape((theta.shape[0], theta.shape[0])) - rVh_1 @ rVh_1.T)
        #cov_cand = infos[k]['sig2'] * (r_2.reshape((theta.shape[0], theta_cand.shape[0])) - rVh_1 @ rVh_4.T)
     

    #print(predmean_ref.shape)
    #print(predvars_cand.shape)
    #print(cov_cand_3D.shape)
    #print(cov_ref_3D.shape)
    # calculate predictive mean and variance
    #predinfo['mean'] = np.full((x.shape[0], int(theta.shape[0]/x.shape[0])), np.nan)
    #predinfo['varcand'] = np.full((theta_cand.shape[0], theta_cand.shape[0]), np.nan)

    pctscale     = (fitinfo['pcti'].T * fitinfo['standardpcinfo']['scale']).T
    Smat3D       = cov_ref_3D*(pctscale[:, :] ** 2)
    cov_cand_3D  = cov_cand_3D*(pctscale[:, :] ** 2)
    cov_cand_3DT = np.transpose(cov_cand_3D, (0, 2, 1)) 
    

    predinfo['mean'] = ((predmean_ref @ pctscale.T) +
                                    fitinfo['standardpcinfo']['offset']).T
    predinfo['mean'] = predinfo['mean'].reshape(n_ref, n_x)
    predinfo['varcand'] = ((fitinfo['standardpcinfo']['extravar'][:] +
                                    predvars_cand @ (pctscale[:, :] ** 2).T)).T
    
    Phi3D        = (cov_cand_3D * cov_cand_3DT)/predinfo['varcand']
    
    #print(Phi3D.shape)
    #print(predinfo['mean'].shape)
    d     = 1*n_x#x.shape[0]
    #eivar = 0
    obsvar3D  = obsvar.reshape(1, n_x, n_x)
    cov1 = 0.5*(Smat3D + obsvar3D + Phi3D)
    cov2 = Smat3D + obsvar3D - Phi3D
    #cov3 = 0.5*obsvar3D + Smat3D
    #det2 = multiple_determinants(obsvar3D)

    #obs_mult = np.repeat(obs, predinfo['mean'].shape[0], axis=0)
    #print(predinfo['mean'].shape)
    #p2 = multiple_pdfs(obs, predinfo['mean'], cov3)
    #print(predinfo['mean'].shape)
    #print(Smat3D.shape)
    p1 = multiple_pdfs(obs, predinfo['mean'], cov1)
    det1 = multiple_determinants(cov2)
    #eivar = np.sum((1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det2)))*p2) - np.sum((1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det1)))*p1)
    eivar = np.sum((1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det1)))*p1)

    return eivar



def postpred(fitinfo, x, theta, obs, obsvar):


    n_x       = len(x)
    n_tot_ref = theta.shape[0]
    n_ref     = int(n_tot_ref/n_x)
    n_t       = fitinfo['theta'].shape[0]
    
    
    predinfo = {}
    infos = fitinfo['emulist']
    predmean_ref = np.zeros((theta.shape[0], len(infos)))
    predvars_ref = np.zeros((theta.shape[0], len(infos)))

    if predmean_ref.ndim < 1.5:
        predmean_ref = predmean_ref.reshape((1, -1))
        predvars_ref = predvars_ref.reshape((1, -1))

    # n_ref x n_t
    rsave_1 = np.array(np.ones(len(infos)), dtype=object)
    # n_ref x n_ref
    rsave_3 = np.array(np.ones(len(infos)), dtype=object)


    # loop over principal components
    for k in range(0, len(infos)):
        if infos[k]['hypind'] == k:
            # covariance matrix between new theta and thetas from fit.
            rsave_1[k] = __covmat(theta,
                                fitinfo['theta'],
                                infos[k]['hypcov'])

            rsave_3[k] = __covmat(theta,
                                  theta,
                                  infos[k]['hypcov'])


        # adjusted covariance matrix
        r_1 = (1 - infos[k]['nug']) * np.squeeze(rsave_1[infos[k]['hypind']])
        r_3 = (1 - infos[k]['nug']) * np.squeeze(rsave_3[infos[k]['hypind']])

        try:
            rVh_1 = r_1 @ infos[k]['Vh']

            
        except Exception:
            for i in range(0, len(infos)):
                print((i, infos[i]['hypind']))
            raise ValueError('Something went wrong with fitted components')


        if rVh_1.ndim < 1.5:
            rVh_1 = rVh_1.reshape((1, -1))
            

        id_row = np.arange(0, n_tot_ref)
        id_col = np.arange(0, n_tot_ref).reshape(n_ref, n_x)
        id_col = np.repeat(id_col, repeats=n_x, axis=0)
        
        r_3_3D = r_3[id_row[:, None], id_col].reshape(n_ref, n_x, n_x)

    
        rVh_1_3d = rVh_1.reshape(n_ref, n_x, n_t)
        rVh_1_3dT = np.transpose(rVh_1_3d, (0, 2, 1))
        cov3D = np.matmul(rVh_1_3d, rVh_1_3dT)
        cov_ref_3D = infos[k]['sig2'] * (r_3_3D - cov3D)


        
        predmean_ref[:, k] = r_1 @ infos[k]['pw']


    # calculate predictive mean and variance
    predinfo['mean'] = np.full((x.shape[0], int(theta.shape[0]/x.shape[0])), np.nan)

    pctscale     = (fitinfo['pcti'].T * fitinfo['standardpcinfo']['scale']).T
    Smat3D       = cov_ref_3D*(pctscale[:, :] ** 2)


    predinfo['mean'] = ((predmean_ref @ pctscale.T) +
                                    fitinfo['standardpcinfo']['offset']).T
    predinfo['mean'] = predinfo['mean'].reshape(n_ref, n_x)


    d     = x.shape[0]

    obsvar3D  = obsvar.reshape(1, n_x, n_x)
    cov1 = 0.5*obsvar3D + Smat3D
    cov2 = Smat3D + obsvar3D
    # cov2 = obsvar3D
    p1 = multiple_pdfs(obs, predinfo['mean'], cov1)
    postmean = multiple_pdfs(obs, predinfo['mean'], cov2)
   
    det1 = multiple_determinants(obsvar3D)
    postvar = (1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det1)))*p1 - postmean**2

    return postmean, postvar