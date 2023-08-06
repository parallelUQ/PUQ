import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred

def pmaxvar(prior_func, prior_func_x, emu, x_emu, theta_mle, x_mesh, th_mesh, synth_info, emubias=None, des=None):

    # nf x d_x
    x_temp = np.array([e['x'] for e in des])

    # 1 x nf
    f_temp = np.array([np.mean(e['feval']) for e in des])[None, :]
    r_temp = [e['rep'] for e in des]
    
    nx_ref = x_mesh.shape[0]
    dx = x_mesh.shape[1]
    nt_ref = th_mesh.shape[0]
    dt = th_mesh.shape[1]
    nf = x_temp.shape[0]

    # Create a candidate list
    n_clist = 100
    #if dt < 2:
    xclist = prior_func_x.rnd(n_clist, None)
    #xclist  = sps.uniform.rvs(0, 1, size=n_clist).reshape(n_clist, dx)
        #xclist  = sps.uniform.rvs(-3, 6, size=n_clist).reshape(n_clist, dx)
   # else:
    #sampling = LHS(xlimits=synth_info.thetalimits[0:2])
    #xclist   = sampling(n_clist)
    
    xclist = np.concatenate((xclist, x_temp))
    ncand = xclist.shape[0]
    # nt_ref x d_t
    theta_ref = 1*th_mesh

    # Get estimate for real data at theta_mle for reference x
    # ncand x (d_x + d_t)
    xt_cand = np.concatenate((xclist, np.repeat(theta_mle, ncand).reshape(ncand, len(theta_mle))), axis=1)
    # 1 x ncand
    y_cand  = emu.predict(x=x_emu, theta=xt_cand).mean()
    # ncand x nf
    f_temp_rep  = np.repeat(f_temp, ncand, axis=0)
    # ncand x (nf + 1)
    f_field_rep = np.concatenate((f_temp_rep, y_cand.T), axis=1)

    xs = [np.concatenate([x_temp, xc.reshape(1, dx)], axis=0) for xc in xclist]

    var_val = np.zeros(len(xclist))
    for xt_id, x_c in enumerate(xclist):
        xdes = xs[xt_id].reshape(nf + 1, dx)
        obsdes = f_field_rep[xt_id, :].reshape(1, nf + 1)
        obsvardes = np.diag(synth_info.realvar(xdes))
        xthmes = [np.concatenate((xdes, np.repeat(theta_mle, nf + 1).reshape(nf + 1, len(theta_mle))), axis=1)]
        xthmes = np.array([m for mesh in xthmes for m in mesh])
        pmeanhat, pvarhat = postpred(emu._info, xdes, xthmes, obsdes, obsvardes)
        var_val[xt_id] = np.sum(pvarhat)
        
  
    maxid = np.argmax(var_val)
    xnew  = xclist[maxid]

    xnew_var = synth_info.realvar(xnew)
    y_temp   = synth_info.genobsdata(xnew, xnew_var)
    des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
            
    print(des)
                
    return des  
