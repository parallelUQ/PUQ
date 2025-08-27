import numpy as np
from scipy.optimize import minimize_scalar
#from distance import distance_cpp

def distance_cpp(X1, X2=None):
    if X2 is None:
        return np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X1**2, axis=1) - 2 * np.dot(X1, X1.T)
    return np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)

def find_corres(X0, X):
    # X0: matrix of unique designs
    # X: matrix of all designs
    # Return: vector associating rows of X with those of X0

    corres = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        for j in range(X0.shape[0]):
            if np.array_equal(X[i, :], X0[j, :]):
                corres[i] = j + 1
                break

    return corres

def find_reps(X, Z, return_Zlist=True, rescale=False, normalize=False, input_bounds=None):

    if np.ndim(X) == 1:
        X = np.expand_dims(X, axis=1)

    if X.shape[0] == 1:
        if return_Zlist:
            return {'X0': X, 'Z0': Z, 'mult': np.array([1]), 'Z': Z, 'Zlist': [Z]}
        return {'X0': X, 'Z0': Z, 'mult': np.array([1]), 'Z': Z}

    if rescale:
        if input_bounds is None:
            input_bounds = np.array([np.min(X, axis=0), np.max(X, axis=0)])
        X = (X - input_bounds[0, :]) @ np.diag(1 / (input_bounds[1, :] - input_bounds[0, :]))

    output_stats = None
    if normalize:
        output_stats = [np.mean(Z), np.var(Z)]
        Z = (Z - output_stats[0]) / np.sqrt(output_stats[1])
    
    X0, indices, inverse_indices = np.unique(X, axis=0, return_index=True, return_inverse=True)
    X0 = X0[np.argsort(indices)]
    corresp = find_corres(X0, X) - 1

    Z0 = [np.mean(Z[corresp == i]) for i in range(len(X0))]
    mult = np.array([np.sum(corresp == i) for i in range(len(X0))])
    Zlist = [Z[corresp == i] for i in range(len(X0))]

    flattened_list = [item for sublist in Zlist for item in sublist]

    if return_Zlist:
        return {'X0': X0, 'Z0': np.array(Z0), 'mult': mult, 'Z': np.array(flattened_list),
                'Zlist': [Z[corres == i] for i in range(len(X0))], 'input_bounds': input_bounds,
                'output_stats': output_stats}
    return {'X0': X0, 'Z0': np.array(Z0), 'mult': mult, 'Z': np.array(flattened_list),
            'input_bounds': input_bounds, 'output_stats': output_stats}



def auto_bounds(X, min_cor=0.01, max_cor=0.5, covtype="Gaussian", p=0.05):
    Xsc = find_reps(X, np.ones(X.shape[0]), rescale=True, return_Zlist=False)  # rescaled distances
    dists = distance_cpp(Xsc['X0'])  # find 2 closest points

    repr_low_dist = np.quantile(dists[np.tril_indices(len(dists), k=-1)], p)
    repr_lar_dist = np.quantile(dists[np.tril_indices(len(dists), k=-1)], 1 - p)

    if covtype == "Gaussian":
        theta_min = -repr_low_dist / np.log(min_cor)
        theta_max = -repr_lar_dist / np.log(max_cor)
        return {'lower': theta_min * (Xsc['input_bounds'][1, :] - Xsc['input_bounds'][0, :]) ** 2,
                'upper': theta_max * (Xsc['input_bounds'][1, :] - Xsc['input_bounds'][0, :]) ** 2}
    # else:
    #     def tmpfun(theta, repr_dist, covtype, value):
    #         cov_gen = cov_gen_matrix(np.sqrt(repr_dist / X.shape[1]), np.zeros((X.shape[1], X.shape[1])), covtype, theta)
    #         return cov_gen - value

    #     theta_min = minimize_scalar(tmpfun, bounds=(np.sqrt(np.finfo(float).eps), 100),
    #                                 args=(repr_low_dist, covtype, min_cor),
    #                                 method='bounded', tol=np.sqrt(np.finfo(float).eps)).x
    #     if theta_min < np.sqrt(np.finfo(float).eps):
    #         warnings.warn("The automatic selection of lengthscales bounds was not successful. Perhaps provide lower and upper values.")
    #         theta_min = 1e-2

    #     theta_max = minimize_scalar(tmpfun, bounds=(np.sqrt(np.finfo(float).eps), 100),
    #                                 args=(repr_lar_dist, covtype, max_cor),
    #                                 method='bounded', tol=np.sqrt(np.finfo(float).eps)).x
    #     if theta_max < np.sqrt(np.finfo(float).eps):
    #         theta_max = 5

    #     return {'lower': theta_min * (Xsc['inputBounds'][1, :] - Xsc['inputBounds'][0, :]),
    #             'upper': max(1, theta_max) * (Xsc['inputBounds'][1, :] - Xsc['inputBounds'][0, :])}
    
    


def trace_sym(A, B):
    n = A.shape[0]
    trace = 0.0

    for i in range(n):
        for j in range(i + 1):
            trace += A[i, j] * B[i, j]

    return trace
