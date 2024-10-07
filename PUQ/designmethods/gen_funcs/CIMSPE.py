import numpy as np
from PUQ.surrogatemethods.PCGPexp import imspe_acq
from smt.sampling_methods import LHS


def imspe(
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
    prior_func_t,
    thetatest=None,
    x_ref=None,
    theta_ref=None,
    posttest=None,
    type_init=None,
    synth_info=None,
    theta_mle=None,
):

    # Create a candidate list
    sampling = LHS(xlimits=thetalimits)
    clist = sampling(1500)

    # Create grid
    sampling = LHS(xlimits=thetalimits)
    mesh_grid = sampling(1500)

    dx = x_ref.shape[1]
    dt = theta_ref.shape[1]

    imspe_val = np.zeros(len(clist))
    for xt_id, x_c in enumerate(clist):
        xt_cand = x_c.reshape(1, dx + dt)
        imspe_val[xt_id] = imspe_acq(emu._info, mesh_grid, xt_cand)

    maxid = np.argmax(imspe_val)
    xnew = clist[maxid].reshape(1, dx + dt)

    return xnew
