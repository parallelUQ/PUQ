"""
Set of functions to fit a homoskedastic GP to each dimension of output data
"""

import numpy as np
from PUQ.surrogate import emulator


def fit(
    fitinfo,
    x,
    theta,
    f,
    lower=None,
    upper=None,
    known={},
    noiseControl={"g_bounds": [np.sqrt(np.finfo(float).eps), 100]},
    init={},
    covtype="Gaussian",
    maxit=100,
    eps=np.sqrt(np.finfo(float).eps),
    settings={"return.Ki": True, "factr": 1e7},
    **kwargs
):
    r"""
    Fit a homoskedastic GP to each dimension of output data.

    Parameters
    ----------
    fitinfo: dictionary/class of results
    x: ndarray
        nxd design matrix (must have at least one column)
    f: ndarray
        output array for training. One GP is trained for each output column.
    theta: not used
    lower,upper : ndarray_like
        optional bounds for the ``theta`` parameter (see :func: covariance_functions.cov_gen for the exact parameterization).
        In the multivariate case, it is possible to give vectors for bounds (resp. scalars) for anisotropy (resp. isotropy)
    noiseControl : dict
        dict with element:
            - ``g_bounds`` vector providing minimal and maximal noise to signal ratio (default to ``(sqrt(MACHINE_DOUBLE_EPS), 100)``).
    settings : dict
                dict for options about the general modeling procedure, with elements:
                    - ``return_Ki`` boolean to include the inverse covariance matrix in the object for further use (e.g., prediction).
                    - ``factr`` (default to 1e7) and ``pgtol`` are available to be passed to `options` for L-BFGS-B in :func: ``scipy.optimize.minimize``.
    eps : float
        jitter used in the inversion of the covariance matrix for numerical stability
    known : dict
        optional dict of known parameters (e.g. ``beta0``, ``theta``, ``g``)
    init :  dict
        optional lists of starting values for mle optimization:
            - ``theta_init`` initial value of the theta parameters to be optimized over (default to 10% of the range determined with ``lower`` and ``upper``)
            - ``g_init`` vector of nugget parameter to be optimized over
    covtype : str
                covariance kernel type, either ``'Gaussian'``, ``'Matern5_2'`` or ``'Matern3_2'``, see :func: ``~covariance_functions.cov_gen``
    maxit : int
            maximum number of iterations for `L-BFGS-B` of :func: ``scipy.optimize.minimize`` dedicated to maximum likelihood optimization

    Returns
    -------
    None, but fitinfo is updated with maximum likelihood estimates. Individual emulators are accessed via fitinfo["emulist"]

    """

    numGPs = f.shape[1]
    emulist = [dict() for x in range(0, numGPs)]
    for i in range(numGPs):
        emu = emulator(
            x=x,
            theta=np.array([[i]]),
            f=f[:, i : i + 1],
            method="homGP",
            args={
                "noiseControl": noiseControl,
                "lower": lower,
                "upper": upper,
                "settings": settings,
                "init": init,
                "known": known,
                "covtype": covtype,
                "maxit": maxit,
                "eps": eps,
            },
        )
        emulist[i] = emu
    fitinfo["f"] = f
    fitinfo["emulist"] = emulist
    fitinfo["numGPs"] = numGPs
    return


def predict(predinfo, fitinfo, x, theta, thetaprime, **kws):
    r"""
    Wrapper method for homGP.predict

    Parameters
    ----------
    predinfo: dict
        (empty) dictionary that will hold prediction results
    fitinfo: dict
        dictionary with hetgpy.homGP-trained hyperparameters and inverse covariance matrices. fitinfo is converted back into a hetgpy.homGP object for prediction
    x: ndarray
        nxd numpy array for prediction. Must match same number of columns as supplied to `fitinfo["X0"]`
    theta: ndarray
        Deprecated, but used to specify output dimension
    thetaprime: ndarray
        nxd numpy array for calculating covariance matrix
    kwargs: dict
        additional keyword arguments passed to hetgpy.homGP.predict

    Returns
    -------
    None, but predinfo is populated with `mean`, `variance`, and `covmat` fields.
    `mean` and `variance` are stored as ndarrays, with each column corresponding to the prediction for the ith output column.
    `covmat` is stored as a list of matrices with the ith element corresponding to the ith output column
    """
    numGPs = fitinfo["numGPs"]
    emulist = [dict() for x in range(0, numGPs)]

    # instantiate outputs
    nr, nc = (x.shape[0], numGPs)
    for key in ("mean", "var", "nugs"):
        predinfo[key] = np.zeros(shape=(nc, nr), dtype=float)

    if thetaprime is None:
        predinfo["covmat"] = np.zeros(shape=(nc, nr, nr), dtype=float)
    else:
        predinfo["covmat"] = np.zeros(shape=(nc, nr, thetaprime.shape[0]), dtype=float)

    for i in range(numGPs):
        preds = fitinfo["emulist"][i].predict(x=x, thetaprime=thetaprime)
        predinfo["mean"][i, :] = preds._info["mean"]
        predinfo["var"][i, :] = preds._info["var"]
        predinfo["nugs"][i, :] = preds._info["nugs"]
        predinfo["covmat"][i, :, :] = preds._info["covmat"]

    predinfo["S"] = np.full((numGPs, numGPs, x.shape[0]), np.nan)
    for i in range(0, x.shape[0]):
        C = np.diag(predinfo["var"][:, i])
        predinfo["S"][:, :, i] = C

    return


def update(fitinfo, x, Y=None, **kwargs):
    r"""
    Update function for homGP

    Parameters
    ----------
    fitinfo: dictionary that contains the fit information for a hetgpy.homGP object
    x: array of new design locations
    Y: new response. If None, then a kriging believer approach is used to impute the predicted mean at the design location
    kwargs: key-value pairs that get passed to hetgpy.homGP.update.
            Must be one of: ginit, lower, upper, noiseControl, settings, known, maxit

    Returns
    -------
    None, but fitinfo is updated in place and individual emulator are accessed via fitinfo["emulist"]
    """
    numGPs = fitinfo["numGPs"]
    for i in range(numGPs):
        emu = fitinfo["emulist"][i]
        if Y is not None:
            Yi = Y[:, i]
        else:
            Yi = None
        emu.update(x=x, Y=Yi, **kwargs)
    return
