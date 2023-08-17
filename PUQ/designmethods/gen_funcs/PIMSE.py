import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred

def pimse(prior_func, prior_func_x, emu, x_emu, theta_mle, x_mesh, th_mesh, synth_info, emubias=None, des=None):

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
    xclist = prior_func_x.rnd(n_clist, None)
    #xclist = np.concatenate((xclist, x_temp))
    ncand = xclist.shape[0]
    
    # nt_ref x d_t
    theta_ref = 1*th_mesh

    var_val = np.zeros(len(xclist))
    for xt_id, x_c in enumerate(xclist):
        xdes = x_c.reshape(1, dx)

        xthmes = [np.concatenate((xdes, np.repeat(th, 1).reshape(1, len(th))), axis=1) for th in th_mesh]
        xthmes = np.array([m for mesh in xthmes for m in mesh])
        
        #xthmes = [np.concatenate((xdes, np.repeat(theta_mle, 1).reshape(1, len(theta_mle))), axis=1)]
        #xthmes = np.array([m for mesh in xthmes for m in mesh])
        
        pmeanhat = emu.predict(x=x_emu, theta=xthmes).mean() 
        pvarhat = emu.predict(x=x_emu, theta=xthmes).var() 
        var_val[xt_id] = np.sum(pvarhat)
        
    plt.scatter(xclist, var_val)
    plt.yscale('log')
    plt.show()
    maxid = np.argmax(var_val)
    xnew  = xclist[maxid]

    xnew_var = synth_info.realvar(xnew)
    y_temp   = synth_info.genobsdata(xnew, xnew_var)
    des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
            
    print(des)
                
    return des  