import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred

def peivarinitial(prior_func, prior_func_x, emu, x_emu, theta_mle, x_mesh, th_mesh, synth_info, emubias=None, des=None):


    nx_ref = x_mesh.shape[0]
    dx = x_mesh.shape[1]
    nt_ref = th_mesh.shape[0]
    dt = th_mesh.shape[1]
    nf = 0

    # Create a candidate list
    n_clist = 100
    xclist = prior_func_x.rnd(n_clist, None)

    # nx_ref x d_x
    x_ref     = 1*x_mesh 
    # nt_ref x d_t
    theta_ref = 1*th_mesh

    eivar_val = np.zeros(len(xclist))
    for th in th_mesh:

        # Get estimate for real data at theta_mle for reference x
        # nx_ref x (d_x + d_t)
        xt_ref = np.concatenate((x_ref, np.repeat(th, nx_ref).reshape(nx_ref, len(th))), axis=1)
        # 1 x nx_ref
        y_ref  = emu.predict(x=x_emu, theta=xt_ref).mean()


        theta_mle = th.reshape(1, dt)
        
        #for th_id, th in enumerate(th_mesh):
        ts = [np.repeat(theta_mle.reshape(1, dt), 1, axis=0)]
        # Construct obsvar
        obsvar_cand = synth_info.realvar(xt_ref[:, 0:dx])
        obsvar3D = np.zeros(shape=(nx_ref, nf+1, nf+1)) 
        for i in range(nx_ref):
            obsvar3D[i, :, :] = obsvar_cand[i]

        n_x = nf + 1
        
        Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, xt_ref, y_ref, obsvar3D)

        for xt_id, x_c in enumerate(xclist):
            x_cand = x_c.reshape(1, dx)
            xt_cand = np.concatenate([x_cand, theta_mle], axis=1)
            eivar_val[xt_id] += postphimat(emu._info, n_x, xt_ref, y_ref, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean)


    maxid = np.argmax(eivar_val)
    xnew  = xclist[maxid].reshape(1, dx)
    print(xnew.shape)
    xnew_var = synth_info.realvar(xnew)
    y_temp   = synth_info.genobsdata(xnew, xnew_var) 
    des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
            
    print(des)
                
    return des  

def pmaxvarinitial(prior_func, prior_func_x, emu, x_emu, theta_mle, x_mesh, th_mesh, synth_info, emubias=None, des=None):


    nx_ref = x_mesh.shape[0]
    dx = x_mesh.shape[1]
    nt_ref = th_mesh.shape[0]
    dt = th_mesh.shape[1]
    nf = 0

    # Create a candidate list
    n_clist = 100
    xclist = prior_func_x.rnd(n_clist, None)

    # nx_ref x d_x
    x_ref     = 1*x_mesh 
    # nt_ref x d_t
    theta_ref = 1*th_mesh

    eivar_val = np.zeros(len(xclist))
    for xt_id, x_c in enumerate(xclist):
        x_cand = x_c.reshape(1, dx)
        xt_cand = [np.concatenate([x_cand, th.reshape(1, len(th))], axis=1) for th in th_mesh]
        xt_cand = np.array([m for mesh in xt_cand for m in mesh])
  
        means = emu.predict(x=x_emu, theta=xt_cand).mean()
        
        
        eivar_val[xt_id] = np.var(means)

    plt.scatter(xclist, eivar_val)
    plt.show()
    maxid = np.argmax(eivar_val)
    xnew  = xclist[maxid]
    xnew_var = synth_info.realvar(xnew)
    
    for i in range(2):
        y_temp   = synth_info.genobsdata(xnew, xnew_var) 
        des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
            
    print(des)
                
    return des  