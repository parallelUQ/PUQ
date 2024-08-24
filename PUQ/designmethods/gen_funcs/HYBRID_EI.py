import numpy as np
import scipy
from PUQ.designmethods.gen_funcs.EI import eifunc
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    compute_postvar,
    compute_eivar,
    multiple_pdfs,
    get_emuvar,
)
from smt.sampling_methods import LHS


def eivarfunc(clist, x, real_x, thetatest, refsize, obs, obsvar3d, emu, priortest):
    ids = np.random.choice(len(thetatest), refsize, replace=False)
    thetaref = thetatest[ids, :]
    emupredict = emu.predict(x, thetaref)
    emumean = emupredict.mean()
    emumeanT = emumean.T
    emuvar, is_cov = get_emuvar(emupredict)
    emuvarT = emuvar.transpose(1, 0, 2)
    var_obsvar1 = emuvarT + obsvar3d

    # Get the n_ref x d x d x n_cand phi matrix
    emuphi4d = emu.acquisition(x=x, theta1=thetaref, theta2=clist)
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
            priortest,
        )
        acq_func.append(-1 * posteivar)

    return acq_func


def hybrid_ei(
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
    believer=0,
    candsize=100,
    refsize=100,
):

    # Update emulator for uncompleted jobs.
    idnan = np.isnan(fevals).any(axis=0).flatten()
    fevals_c = fevals[:, ~idnan]

    theta_uc = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean())

    n_x, p = x.shape[0], theta.shape[1]
    theta_acq = []
    obsvar3d = obsvar.reshape(1, n_x, n_x)

    theta_acq, f_acq = [], []
    is_ei = True if (len(theta) % 2) == 0 else False

    # Best error
    error = np.sum(np.abs(obs - fevals_c.T), axis=1)
    delta = np.abs(obs - fevals_c.T)[np.argmin(error)]
    liar = np.mean(fevals_c)
    for i in range(n):
        clist = prior_func.rnd(candsize, None)

        if is_ei:
            acq_val = eifunc(clist, x, obs, obsvar, emu, delta)
            is_ei = False
        else:
            acq_val = eivarfunc(
                clist, x, real_x, thetatest, refsize, obs, obsvar3d, emu, posttest
            )
            is_ei = True

        if n == 1:
            idc = np.argmax(acq_val)
            ctheta = clist[idc, :].reshape((1, p))
            theta_acq.append(ctheta)
            break
        else:
            if believer == 1:
                idc = np.argmax(acq_val)
                ctheta = clist[idc, :].reshape((1, p))

                # Kriging believer strategy
                fevalsnew = emu.predict(x=x, theta=ctheta)
                emu.update(theta=ctheta, f=fevalsnew.mean())

            elif believer == 2:
                idc = np.argmax(acq_val)
                ctheta = clist[idc, :].reshape((1, p))

                # Constant liar strategy
                fevalsnew = liar.reshape(1, 1)
                emu.update(theta=ctheta, f=fevalsnew)

            theta_acq.append(ctheta)

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq
