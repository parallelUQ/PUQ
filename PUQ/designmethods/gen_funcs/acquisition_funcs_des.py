import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps

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
          thetamesh=None, 
          posttest=None,
          type_init=None):
    
    # Update emulator for uncompleted jobs.
    idnan    = np.isnan(fevals).any(axis=0).flatten()
    theta_uc = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean()) 

    p    = theta.shape[1]
    
    # Create a candidate list
    n_clist = 500
    if type_init == 'LHS':
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
        clist   = prior_func.rnd(n_clist, None)
    
        # n_clist  = 50*9
        # sampling = LHS(xlimits=thetalimits[0:2])
        # t_unif   = sampling(50)
        # x_unif   = np.repeat(x, 50, axis=0)
        # t_unif_rep = np.tile(t_unif, (9,1))
        # clist = np.concatenate((x_unif, t_unif_rep), axis=1) 


    n_x = x.shape[0]
    #print(n_x)
    thetatest = np.array([np.concatenate([xc, th]) for th in thetamesh for xc in x ])
    #print(thetatest)
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, thetatest, obs, obsvar)
    #print('here')
    eivar_val = np.zeros(n_clist)
    for xt_id, xt_c in enumerate(clist):
        eivar_val[xt_id] = postphimat(emu._info, n_x, thetatest, obs, obsvar, xt_c.reshape(1, p), Smat3D, rVh_1_3d, pred_mean)

    th_cand = clist[np.argmax(eivar_val), :].reshape(1, p)
    print(th_cand)
    return th_cand  



def eivar_des_updated(prior_func, emu, x_emu, theta_mle, x_mesh, th_mesh, synth_info, emubias=None, des=None):

    # nf x d_x
    x_temp = np.array([e['x'] for e in des])[:, None]
    print(x_temp)
    # 1 x nf
    f_temp = np.array([np.mean(e['feval']) for e in des])[None, :]
    r_temp = [e['rep'] for e in des]
    
    # Create a candidate list
    n_clist = 100
    #clist   = prior_func.rnd(n_clist, None)
    xclist  = sps.uniform.rvs(0, 1, size=n_clist) # clist[:, 0]

    nx_ref = x_mesh.shape[0]
    dx = x_mesh.shape[1]
    nt_ref = th_mesh.shape[0]
    dt = th_mesh.shape[1]
    
    nf = x_temp.shape[0]
    
    # nx_ref x d_x
    x_ref     = 1*x_mesh 
    # nt_ref x d_t
    theta_ref = 1*th_mesh

    # Get estimate for real data at theta_mle for reference x
    # nx_ref x (d_x + d_t)
    xt_ref = np.concatenate((x_ref, np.repeat(theta_mle, nx_ref)[:, None]), axis=1)
    # 1 x nx_ref
    y_ref  = emu.predict(x=x_emu, theta=xt_ref).mean()
    # nx_ref x nf
    f_temp_rep  = np.repeat(f_temp, nx_ref, axis=0)
    # nx_ref x (nf + 1)
    f_field_rep = np.concatenate((f_temp_rep, y_ref.T), axis=1)
    # (nx_ref + nt_ref) x (nf + 1)
    f_field_all = np.repeat(f_field_rep, nt_ref, axis=0)

    print(x_ref)
    print(x_temp.shape)
    xs = [np.concatenate([x_temp, xc[:, None]], axis=0) for xc in x_ref]
    ts = [np.repeat(th[:, None], nf + 1, axis=0) for th in theta_ref]
    mesh_grid = [np.concatenate([xc, th], axis=1).tolist() for xc in xs for th in ts]
    mesh_grid = np.array([m for mesh in mesh_grid for m in mesh])
    
    
    obsvar_field  = np.diag(np.repeat(synth_info.sigma2, f_field_rep.shape[1]))
    n_x = obsvar_field.shape[0]
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, mesh_grid, f_field_all, obsvar_field)

    eivar_val = np.zeros(len(xclist))
    theta_mle = theta_mle.reshape(1, dt)
    for xt_id, x_c in enumerate(xclist):
        x_cand = x_c.reshape(1, dx)
        xt_cand = np.concatenate([x_cand, theta_mle], axis=1)
        eivar_val[xt_id] = postphimat(emu._info, n_x, mesh_grid, f_field_all, obsvar_field, xt_cand, Smat3D, rVh_1_3d, pred_mean)

    maxid = np.argmax(eivar_val)
    xnew  = xclist[maxid]
    y_temp    = synth_info.function(xnew, synth_info.true_theta) + np.random.normal(0, np.sqrt(synth_info.sigma2), 1)
    des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
        
    print(des)
                
    return des  


