import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred
   
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
    type_init = 'CMB'

    xuniq = np.unique(x, axis=0)

    print(xuniq)
    # Create a candidate list
    if type_init == 'LHS':
        n_clist = 500
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    elif type_init == 'RND':
        n_clist = 500
        clist = prior_func.rnd(n_clist, None)
    elif type_init == 'CMB':
        n0 = 50
        n_clist  = n0*len(xuniq)
        t_unif = prior_func_t.rnd(n0, None)
        clist1 = np.array([np.concatenate([xc, th]) for th in t_unif for xc in xuniq])
        clist2 = prior_func.rnd(n_clist, None)
        
        clist = np.concatenate((clist1, clist2), axis=0)
        n_clist += n_clist
    else:
        n0 = 50
        n_clist  = n0*len(xuniq)
        t_unif   = prior_func_t.rnd(n0, None)
        clist = np.array([np.concatenate([xc, th]) for th in t_unif for xc in xuniq])


    n_x = x.shape[0]
    thetatest = np.array([np.concatenate([xc, th]) for th in thetamesh for xc in x])
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, thetatest, obs, obsvar)
    eivar_val = np.zeros(n_clist)
    for xt_id, xt_c in enumerate(clist):
        eivar_val[xt_id] = postphimat(emu._info, n_x, thetatest, obs, obsvar, xt_c.reshape(1, p), Smat3D, rVh_1_3d, pred_mean)

    th_cand = clist[np.argmax(eivar_val), :].reshape(1, p)

    return th_cand 