import numpy as np
from numpy.linalg import inv, det

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