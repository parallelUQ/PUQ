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
    believer=None,
    candsize=None,
    refsize=None,
):

    theta_acq = prior_func.rnd(n, None)
    return theta_acq
