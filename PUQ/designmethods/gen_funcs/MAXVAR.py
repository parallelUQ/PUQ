import numpy as np
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    compute_postvar,
    multiple_pdfs,
    get_emuvar,
)
from smt.sampling_methods import LHS
import scipy


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
    diags = np.diag(obsvar[real_x, real_x.T])
    d_real = real_x.shape[0]
    coef = (2**d_real) * (np.sqrt(np.pi) ** d_real) * np.sqrt(np.prod(diags))

    # Create a candidate list.
    n_clist = 100 * n
    if type_init == "LHS":
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
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
