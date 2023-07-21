"""Contains supplemental methods for acquisitionn funcs."""

import numpy as np
import scipy.stats as sps


def compute_likelihood(emumean, emuvar, obs, obsvar, is_cov):
    if emumean.shape[0] == 1:
        emuvar = emuvar.reshape(emumean.shape)
        ll = sps.norm.pdf(obs - emumean, 0, np.sqrt(obsvar + emuvar))
    else:
        ll = np.zeros(emumean.shape[1])
        for i in range(emumean.shape[1]):
            mean = emumean[:, i]  # [emumean[0, i], emumean[1, i]]

            if is_cov:
                cov = emuvar[:, i, :] + obsvar
            else:
                cov = np.diag(emuvar[:, i]) + obsvar

            rnd = sps.multivariate_normal(mean=mean, cov=cov)
            ll[i] = rnd.pdf(obs)  # rnd.pdf([obs[0, 0], obs[0, 1]])

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
    part1 = part1 * (1 / coef)

    # part2 = compute_likelihood(emumean, emuvar, obs, obsvar, is_cov)
    part2 = part2**2

    return part1 - part2


def compute_eivar_fig(
    obsvar, summatrix2, summatrix, emuphi, emumean, emuvar, obs, is_cov
):
    rndpdf2 = multiple_pdfs(obs, emumean, summatrix2)
    denum2 = obsvar
    # See Eq. 31
    covmat1 = (summatrix + emuphi) * 0.5
    covmat2 = summatrix - emuphi

    rndpdf = multiple_pdfs(obs, emumean, covmat1)

    denum = multiple_determinants(covmat2)
    part2 = rndpdf / np.sqrt(denum)
    # print(part2.shape)
    return (np.sum(rndpdf2 / np.sqrt(denum2)) - np.sum(part2)) * (
        1 / (2 * np.pi ** (0.5))
    )


def compute_eivar(summatrix, emuphi, emumean, emuvar, obs, is_cov, prioreval):
    # See Eq. 31
    covmat1 = (summatrix + emuphi) * 0.5
    covmat2 = summatrix - emuphi

    rndpdf = multiple_pdfs(obs, emumean, covmat1)

    denum = multiple_determinants(covmat2)
    part2 = rndpdf / np.sqrt(denum)
    # print(part2.shape)
    return -np.sum(part2 * (prioreval**2))


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


def get_emuvar(emupredict):
    is_cov = False
    try:
        emuvar = emupredict.covx()
        is_cov = True
    except Exception:
        emuvar = emupredict.var()

    return emuvar, is_cov


def eivar_sup(clist, theta, theta_test, emu, data_cls):
    real_x = data_cls.real_x
    obs = data_cls.real_data
    obsvar = data_cls.obsvar
    obsvar3d = obsvar.reshape(1, 1, 1)
    x = data_cls.x
    emupred_test = emu.predict(x=data_cls.x, theta=theta_test)
    emumean = emupred_test.mean()
    emuvar, is_cov = get_emuvar(emupred_test)
    emumeanT = emumean.T
    emuvarT = emuvar.transpose(1, 0, 2)
    var_obsvar1 = emuvarT + obsvar3d
    var_obsvar2 = emuvarT + 0.5 * obsvar3d

    # Get the n_ref x d x d x n_cand phi matrix
    emuphi4d = emu.acquisition(x=x, theta1=theta_test, theta2=clist)
    acq_func = []

    # Pass over all the candidates
    for c_id in range(len(clist)):
        posteivar = compute_eivar_fig(
            obsvar,
            var_obsvar2[:, real_x, real_x.T],
            var_obsvar1[:, real_x, real_x.T],
            emuphi4d[:, real_x, real_x.T, c_id],
            emumeanT[:, real_x.flatten()],
            emuvar[real_x, :, real_x.T],
            obs,
            is_cov,
        )
        acq_func.append(posteivar)

    return acq_func
