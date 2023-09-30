import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat
from smt.sampling_methods import LHS
from numpy.random import rand
import scipy.stats as sps

def ceivar(n, 
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
          theta_mle=None):
    

    p = theta.shape[1]
    dt = thetamesh.shape[1]
    dx = x.shape[1]
    n_x = x.shape[0]
    
    xuniq = np.unique(x, axis=0)
    if synth_info.data_name == 'covid19':
        clist = construct_candlist_covid(thetalimits, xuniq, prior_func, prior_func_t)
    else:
        clist = construct_candlist(thetalimits, xuniq, prior_func, prior_func_t)

    xt_ref = np.array([np.concatenate([xc, th]) for th in thetamesh for xc in x])
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, xt_ref, obs, obsvar)
    eivar_val = np.zeros(len(clist))
    for xt_id, xt_c in enumerate(clist):
        eivar_val[xt_id] = postphimat(emu._info, n_x, xt_ref, obs, obsvar, xt_c.reshape(1, p), Smat3D, rVh_1_3d, pred_mean)

    th_cand = clist[np.argmax(eivar_val), :].reshape(1, p)

    return th_cand 

def ceivarbias(n, 
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
          emubias=None,
          synth_info=None,
          theta_mle=None,
          unknowncov=None):
    

    p = theta.shape[1]
    dt = thetamesh.shape[1]
    dx = x.shape[1]
    n_x = x.shape[0]
    x_emu = np.arange(0, 1)[:, None ]
    
    bias_mean = emubias.predict(x)
    if unknowncov:
        bias_var = emubias.predictcov(x)
        print(np.round(bias_var, 2))
    else:
        bias_var = 1*obsvar


    xuniq = np.unique(x, axis=0)
    clist = construct_candlist(thetalimits, xuniq, prior_func, prior_func_t)

    thetatest = np.array([np.concatenate([xc, th]) for th in thetamesh for xc in x])
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, thetatest, obs, bias_var)
    eivar_val = np.zeros(len(clist))
    

    for xt_id, xt_c in enumerate(clist):
        eivar_val[xt_id] = postphimat(emu._info, n_x, thetatest, obs-bias_mean, bias_var, xt_c.reshape(1, p), Smat3D, rVh_1_3d, pred_mean)
    th_cand = clist[np.argmax(eivar_val), :].reshape(1, p)

    return th_cand 

def construct_candlist(thetalimits, xuniq, prior_func, prior_func_t):

    n0 = 100
    n_clist = n0*len(xuniq)
    t_unif = prior_func_t.rnd(n0, None)
    clist1 = np.array([np.concatenate([xc, th]) for th in t_unif for xc in xuniq])
    clist2 = prior_func.rnd(n_clist, None)
    clist = np.concatenate((clist1, clist2), axis=0)

    return clist

def construct_candlist_covid(thetalimits, xuniq, prior_func, prior_func_t):

    # 1200 = 100 x 12
    n0 = 100
    t_unif = prior_func_t.rnd(n0, None)
    clist1 = np.array([np.concatenate([xc, th]) for th in t_unif for xc in xuniq])
    
    # 2500 = 50 x 50
    n0 = 50
    xref_sample = np.random.choice(a=222, size=n0, replace=False)[:, None]/221
    t_unif = prior_func_t.rnd(n0, None)
    clist2 = np.array([np.concatenate([xc, th]) for th in t_unif for xc in xref_sample])
    clist = np.concatenate((clist1, clist2), axis=0)
    
    print(clist.shape)
    return clist