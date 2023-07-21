"""
This module contains a class for posterior.
"""
import numpy as np
import importlib
import copy
import warnings


class posterior(object):
    def __init__(self, data_cls=None, emulator=None, args={}):
        self.data = data_cls
        self.emulator = emulator

    def predict(self, theta):
        emu = self.emulator
        data_cls = self.data
        real_x = data_cls.real_x
        obs = data_cls.real_data
        obsvar = data_cls.obsvar
        d, p = real_x.shape[0], theta.shape[1]

        obsvar3d = obsvar.reshape(1, d, d)
        diags = np.diag(obsvar[real_x, real_x.T])
        d_real = real_x.shape[0]
        coef = (2**d_real) * (np.sqrt(np.pi) ** d_real) * np.sqrt(np.prod(diags))

        emupred_test = emu.predict(x=data_cls.x, theta=theta)
        emumean = emupred_test.mean()
        emuvar, is_cov = get_emuvar(emupred_test)
        emumeanT = emumean.T
        emuvarT = emuvar.transpose(1, 0, 2)
        var_obsvar1 = emuvarT + obsvar3d
        var_obsvar2 = emuvarT + 0.5 * obsvar3d
        diags = np.diag(obsvar[real_x, real_x.T])
        coef = (2**d_real) * (np.sqrt(np.pi) ** d_real) * np.sqrt(np.prod(diags))

        postmean = multiple_pdfs(
            obs, emumeanT[:, real_x.flatten()], var_obsvar1[:, real_x, real_x.T]
        )

        postvar = compute_postvar(
            obs,
            emumeanT[:, real_x.flatten()],
            var_obsvar1[:, real_x, real_x.T],
            var_obsvar2[:, real_x, real_x.T],
            coef,
        )

        return postmean, postvar


def get_emuvar(emupredict):
    is_cov = False
    try:
        emuvar = emupredict.covx()
        is_cov = True
    except Exception:
        emuvar = emupredict.var()

    return emuvar, is_cov


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
