import numpy as np
from PUQ.surrogatemethods.PCGPexp import postphi
from smt.sampling_methods import LHS

def eivar_exp(n, 
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
          type_init=None):
    
    # Update emulator for uncompleted jobs.
    idnan    = np.isnan(fevals).any(axis=0).flatten()
    theta_uc = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean()) 
        
    theta_acq = []
    n_x, p    = x.shape[0], theta.shape[1]
    obsvar3d  = obsvar.reshape(1, n_x, n_x)

    # Create a candidate list
    n_clist = 100*n
    if type_init == 'LHS':
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
        clist   = prior_func.rnd(n_clist, None)
        
    eivar_max = -np.inf
    th_max = 0
    for xt_c in clist:
        eivar_val = 0
        for th_r in thetatest:
            xt_ref = np.concatenate((x, np.repeat(th_r, len(x))[:, None]), axis=1)
            eivar_val += postphi(emu._info, x, xt_ref, obs, obsvar, xt_c.reshape(1, 2))
        
        if eivar_val > eivar_max:
            eivar_max = 1*eivar_val
            th_max = 1*xt_c
    
    print(th_max)
    th_cand = th_max.reshape(1, p)
 
    return th_cand  