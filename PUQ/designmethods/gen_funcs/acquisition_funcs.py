"""Contains acquisition functions."""

import numpy as np
from sklearn.metrics import pairwise_distances
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    compute_postvar,
    compute_eivar,
    multiple_pdfs,
    get_emuvar,
)
import scipy


def rnd(
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
):
    theta_acq = prior_func.rnd(n, None)
    return theta_acq


def maxvar(
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
    diags = np.diag(obsvar[real_x, real_x.T])
    d_real = real_x.shape[0]
    coef = (2**d_real) * (np.sqrt(np.pi) ** d_real) * np.sqrt(np.prod(diags))

    # Create a candidate list.
    n_clist = 100 * n
    clist = prior_func.rnd(n_clist, None)

    # Acquire n thetas.
    for i in range(n):
        emupredict = emu.predict(x, clist)
        emumean = emupredict.mean()
        emuvar, is_cov = get_emuvar(emupredict)
        emumeanT = emumean.T
        emuvarT = emuvar.transpose(1, 0, 2)
        var_obsvar1 = emuvarT + obsvar3d
        var_obsvar2 = emuvarT + 0.5 * obsvar3d
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

        acq_func = postvar.copy()
        idc = np.argmax(acq_func)
        ctheta = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
        f_acq.append(postmean[idc])
        clist = np.delete(clist, idc, 0)

        # Kriging believer strategy
        fevalsnew = emu.predict(x=x, theta=ctheta)
        emu.update(theta=ctheta, f=fevalsnew.mean())

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq


def eivar(
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
):
    # Update emulator for uncompleted jobs.
    idnan = np.isnan(fevals).any(axis=0).flatten()
    theta_uc = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean())

    theta_acq = []
    n_x, p = x.shape[0], theta.shape[1]
    obsvar3d = obsvar.reshape(1, n_x, n_x)

    # Create a candidate list
    if n == 1:
        n_clist = 100 * n
    else:
        n_clist = 100 * int(n)  # 100*int(np.ceil(np.log(n)))

    clist = prior_func.rnd(n_clist, None)

    for i in range(n):
        emupredict = emu.predict(x, thetatest)
        emumean = emupredict.mean()
        emumeanT = emumean.T
        emuvar, is_cov = get_emuvar(emupredict)
        emuvarT = emuvar.transpose(1, 0, 2)
        var_obsvar1 = emuvarT + obsvar3d

        # Get the n_ref x d x d x n_cand phi matrix
        emuphi4d = emu.acquisition(x=x, theta1=thetatest, theta2=clist)
        acq_func = []

        # Pass over all the candidates
        for c_id in range(len(clist)):
            posteivar = compute_eivar(
                var_obsvar1[:, real_x, real_x.T],
                emuphi4d[:, real_x, real_x.T, c_id],
                emumeanT[:, real_x.flatten()],
                emuvar[real_x, :, real_x.T],
                obs,
                is_cov,
                posttest,
            )
            acq_func.append(posteivar)

        idc = np.argmin(acq_func)
        ctheta = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
        clist = np.delete(clist, idc, 0)

        # Kriging believer strategy
        fevalsnew = emu.predict(x=x, theta=ctheta)
        emu.update(theta=ctheta, f=fevalsnew.mean())

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq


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


def imse(
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
):
    # Update emulator for uncompleted jobs.
    idnan = np.isnan(fevals).any(axis=0).flatten()
    theta_uc = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean())

    theta_acq = []
    n_x, p = x.shape[0], theta.shape[1]
    obsvar3d = obsvar.reshape(1, n_x, n_x)

    # Create a candidate list
    if n == 1:
        n_clist = 100 * n
    else:
        n_clist = 100 * int(n)  # 100*int(np.ceil(np.log(n)))

    clist = prior_func.rnd(n_clist, None)

    for i in range(n):
        # Get the n_ref x d x d x n_cand phi matrix
        emuphi4d = emu.acquisition(x=x, theta1=thetatest, theta2=clist)
        acq_func = []

        # print(emuphi4d.shape)
        # Pass over all the candidates
        for c_id in range(len(clist)):
            # print(emuphi4d[c_id, :, :, 0])
            posteivar = np.sum(emuphi4d[:, :, :, c_id])
            acq_func.append(posteivar)

        idc = np.argmax(acq_func)
        ctheta = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
        clist = np.delete(clist, idc, 0)

        # Kriging believer strategy
        fevalsnew = emu.predict(x=x, theta=ctheta)
        emu.update(theta=ctheta, f=fevalsnew.mean())

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq


def ei(
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
):
    n_x, p = x.shape[0], theta.shape[1]
    # Create a candidate list
    if n == 1:
        n_clist = 100 * n
    else:
        n_clist = 100 * int(n)  # 100*int(np.ceil(np.log(n)))

    clist = prior_func.rnd(n_clist, None)
    maxpost = np.max(fevals)

    for i in range(n):
        emupredict = emu.predict(np.arange(0, 1)[:, None], clist)
        emumean = emupredict.mean()
        emuvar = emupredict.var()

        zscore = (emumean - maxpost) / np.sqrt(emuvar)

        cdf_cand = scipy.stats.norm.cdf(zscore)
        pdf_cand = scipy.stats.norm.pdf(zscore)
        ei_cand = (emumean - maxpost) * cdf_cand + np.sqrt(emuvar) * pdf_cand
        acqfun = ei_cand

    # print(acqfun)
    # print(acqfun)
    theta_acq = clist[np.argmax(acqfun), :]

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq
