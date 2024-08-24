import numpy as np
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    compute_postvar,
    compute_eivar,
    multiple_pdfs,
    get_emuvar,
)
from smt.sampling_methods import LHS
import scipy


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

    theta_acq = []
    n_x, p = x.shape[0], theta.shape[1]
    obsvar3d = obsvar.reshape(1, n_x, n_x)

    # Create a candidate list
    if n == 1:
        n_clist = 100 * n
    else:
        n_clist = 100 * int(n)  # 100*int(np.ceil(np.log(n)))

    if type_init == "LHS":
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
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
