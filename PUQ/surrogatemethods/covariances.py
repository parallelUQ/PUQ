#from distance import distance_cpp
import numpy as np

### A) Gaussian covariance

## Gaussian covariance function : K = exp(-(X1 - X2)^2/theta)
## @param X1 matrix of design locations
## @param X2 matrix of design locations if covariance is calculated between X1 and X2
## @param theta vector of lengthscale parameters (either of size one if isotropic or of size d if anisotropic)

def distance_cpp(X1, X2=None):
    if X2 is None:
        return np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X1**2, axis=1) - 2 * np.dot(X1, X1.T)
    return np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)

def cov_gen(X1, X2=None, theta=None):

    if len(theta) == 1:
        return np.exp(-distance_cpp(X1, X2) / theta)
    
    if X2 is None:
        #np.exp(-distance_cpp(X1 * np.tile(1 / np.sqrt(theta), (X1.shape[0], len(theta)))))
        return np.exp(-distance_cpp(X1 * (1/np.sqrt(np.repeat(theta[None, :], X1.shape[0], axis=0)))))

    # np.exp(-distance_cpp(X1 * np.repeat(1/np.sqrt(theta), np.repeat(X1.shape[0], len(theta))),
    #                              X2 * np.repeat(1/np.sqrt(theta), np.repeat(X2.shape[0], len(theta)))))

    return np.exp(-distance_cpp(X1 * (1/np.sqrt(np.repeat(theta[None, :], X1.shape[0], axis=0))),
                                 X2 * (1/np.sqrt(np.repeat(theta[None, :], X2.shape[0], axis=0)))))



def partial_cov_gen(X1, theta, type="Gaussian", arg=None, X2=None, **kwargs):

    if X2 is None:
        if type == "Gaussian":
            if arg == "theta_k":
                return partial_d_C_Gaussian_dtheta_k(X1=X1, theta=theta, **kwargs)
            if arg == "k_theta_g":
                return partial_d_Cg_Gaussian_d_k_theta_g(X1=X1, theta=theta, **kwargs)
            if arg == "X_i_j":
                return partial_d_C_Gaussian_dX_i_j(X1=X1, theta=theta, **kwargs)


def partial_d_C_Gaussian_dtheta_k(X1, theta):
    return distance_cpp(X1) / theta**2

def partial_d_k_Gaussian_dtheta_k(X1, X2, theta):
    return distance_cpp(X1, X2) / theta**2

def partial_d_Cg_Gaussian_d_k_theta_g(X1, theta, k_theta_g):
    # 1-dimensional/isotropic case
    if len(theta) == 1:
        return distance_cpp(X1) / (theta * k_theta_g**2)
    
    return distance_cpp(X1 * (1/np.sqrt(np.repeat(theta[None, :], X1.shape[0], axis=0))))/k_theta_g**2 #distance_cpp(X1 * np.tile(1 / np.sqrt(theta), (X1.shape[0], len(theta)))) / k_theta_g**2

