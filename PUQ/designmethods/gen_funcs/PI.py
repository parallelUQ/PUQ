import numpy as np
import scipy


def pifunc(clist, x, obs, obsvar, emu, delta):
    pi_val = np.zeros(len(clist))
    pp = emu.predict(x, clist)
    ppmean = pp.mean()
    ppvar = pp.var()

    for l in range(ppmean.shape[0]):
        diff = obs[0, l] - ppmean[l, :]
        sumvar = np.diag(obsvar)[l] + ppvar[l, :]

        part1 = scipy.stats.norm.cdf((delta[l] - diff) / np.sqrt(sumvar), 0, 1)
        part2 = scipy.stats.norm.cdf((-delta[l] - diff) / np.sqrt(sumvar), 0, 1)
        pi_val += part1 - part2

    return pi_val


def pi(
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

    p = theta.shape[1]
    theta_acq, f_acq = [], []

    # Best error
    error = np.sum(np.abs(obs - fevals_c.T), axis=1)
    delta = np.abs(obs - fevals_c.T)[np.argmin(error)]
    liar = np.mean(fevals_c)

    for i in range(n):
        clist = prior_func.rnd(candsize, None)

        acq_val = pifunc(clist, x, obs, obsvar, emu, delta)
        if believer == 0:
            idc = np.argsort(acq_val)[::-1][:n]
            ctheta = clist[idc, :].reshape((n, p))
            theta_acq.append(ctheta)
            break

        elif believer == 1:
            idc = np.argmax(acq_val)
            ctheta = clist[idc, :].reshape((1, p))
            theta_acq.append(ctheta)
            clist = np.delete(clist, idc, 0)

            # Kriging believer strategy
            fevalsnew = emu.predict(x=x, theta=ctheta)
            emu.update(theta=ctheta, f=fevalsnew.mean())
        elif believer == 2:
            idc = np.argmax(acq_val)
            ctheta = clist[idc, :].reshape((1, p))
            theta_acq.append(ctheta)
            clist = np.delete(clist, idc, 0)

            # Kriging believer strategy
            fevalsnew = liar.reshape(1, 1)
            emu.update(theta=ctheta, f=fevalsnew)

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq
