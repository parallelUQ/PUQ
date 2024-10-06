import numpy as np
from sklearn.metrics import pairwise_distances
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    compute_postvar,
    compute_eivar,
    multiple_pdfs,
    get_emuvar,
)
from smt.sampling_methods import LHS
import scipy


def maxexp(
    n,
    x,
    real_x,
    emu,
    theta,
    fevals,
    obs,
    obsvar,
    thetalimits,
    prior_func,
    thetatest=None,
    posttest=None,
    type_init=None,
    believer=None,
    candsize=None,
    refsize=None,
):

    # Update emulator for uncompleted jobs.
    idnan = np.isnan(fevals).any(axis=0).flatten()
    theta_uc = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean())

    theta_acq, f_acq = [], []
    d, p = x.shape[0], theta.shape[1]
    obsvar3d = obsvar.reshape(1, d, d)

    # Create a candidate list.
    n_clist = 100 * n
    if type_init == "LHS":
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
        clist = prior_func.rnd(n_clist, None)

    theta_sd = (theta - thetalimits[:, 0]) / (thetalimits[:, 1] - thetalimits[:, 0])
    theta_cand_sd = (clist - thetalimits[:, 0]) / (
        thetalimits[:, 1] - thetalimits[:, 0]
    )

    for j in range(n):

        emupredict = emu.predict(x, clist)
        emumean = emupredict.mean()
        emuvar, is_cov = get_emuvar(emupredict)
        emumeanT = emumean.T
        emuvarT = emuvar.transpose(1, 0, 2)
        var_obsvar1 = emuvarT + obsvar3d
        postmean = multiple_pdfs(
            obs, emumeanT[:, real_x.flatten()], var_obsvar1[:, real_x, real_x.T]
        )
        logpostmean = np.log(postmean)

        # Diversity term
        d = pairwise_distances(theta_sd, theta_cand_sd, metric="euclidean")
        min_dist = np.min(d, axis=0)

        # Acquisition function
        acq_func = logpostmean + np.log(min_dist)

        # New theta
        idc = np.argmax(acq_func)
        theta_sd = np.concatenate((theta_sd, theta_cand_sd[idc][None, :]), axis=0)
        ctheta = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
        f_acq.append(postmean[idc])
        clist = np.delete(clist, idc, 0)
        theta_cand_sd = np.delete(theta_cand_sd, idc, 0)

        # Kriging believer strategy
        fevalsnew = emu.predict(x=x, theta=ctheta)
        emu.update(theta=ctheta, f=fevalsnew.mean())

    theta_acq = np.array(theta_acq).reshape((n, p))
    return theta_acq
