import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat
from smt.sampling_methods import LHS
from numpy.random import rand
import scipy.stats as sps


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
    prior_func_t,
    thetatest=None,
    xmesh=None,
    thetamesh=None,
    posttest=None,
    type_init=None,
    synth_info=None,
    theta_mle=None,
):
    """
    Smat3D    : ncand x nfield x nfield (3D) Emulator Covariance matrix
    pred_mean : ncand x nfield (2D) Emulator Mean matrix
    rVh_1_3d  : ncand x nfield x nacquired (3D) matrix
    """

    p = theta.shape[1]
    dt = thetamesh.shape[1]
    dx = x.shape[1]
    n_x = x.shape[0]

    xuniq = np.unique(x, axis=0)
    # if synth_info.data_name == 'covid19':
    #     clist = construct_candlist_covid(thetalimits, xuniq, prior_func, prior_func_t)
    # else:
    clist = construct_candlist(thetalimits, xuniq, prior_func, prior_func_t)

    emupred = emu.predict(x=np.arange(0, 1)[:, None], theta=clist)
    emuvar = emupred.var()
    th_cand = clist[np.argmax(emuvar.flatten()), :].reshape(1, p)
    return th_cand


def construct_candlist(thetalimits, xuniq, prior_func, prior_func_t):

    # Create a candidate list
    sampling = LHS(xlimits=thetalimits)
    clist = sampling(1500)
    return clist
