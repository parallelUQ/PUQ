import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred

def peivareff(prior_func, prior_func_x, emu, x_emu, theta_mle, x_mesh, th_mesh, synth_info, emubias=None, des=None):

    # nf x d_x
    x_temp = np.array([e['x'] for e in des])#[:, None]

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
    #n_clist = 100
    #if dt < 2:
        #xclist  = sps.uniform.rvs(0, 1, size=n_clist).reshape(n_clist, dx)
        #xclist  = sps.uniform.rvs(-3, 6, size=n_clist).reshape(n_clist, dx)
    #else:
    #sampling = LHS(xlimits=synth_info.thetalimits[0:2])
    #xclist   = sampling(n_clist)
    
    xclist = np.concatenate((xclist, x_temp))
    #print(xclist)
    # nx_ref x d_x
    x_ref     = 1*x_mesh 
    # nt_ref x d_t
    theta_ref = 1*th_mesh

    # Get estimate for real data at theta_mle for reference x
    # nx_ref x (d_x + d_t)
    xt_ref = np.concatenate((x_ref, np.repeat(theta_mle, nx_ref).reshape(nx_ref, len(theta_mle))), axis=1)

    # 1 x nx_ref
    y_ref  = emu.predict(x=x_emu, theta=xt_ref).mean()
    # nx_ref x nf
    f_temp_rep  = np.repeat(f_temp, nx_ref, axis=0)
    # nx_ref x (nf + 1)
    f_field_rep = np.concatenate((f_temp_rep, y_ref.T), axis=1)
    # (nx_ref + nt_ref) x (nf + 1)
    #f_field_all = np.repeat(f_field_rep, nt_ref, axis=0)
    

    xs = [np.concatenate([x_temp, xc.reshape(1, dx)], axis=0) for xc in x_ref]


    eivar_val = np.zeros(len(xclist))
    theta_mle = theta_mle.reshape(1, dt)
    
    #for th_id, th in enumerate(th_mesh):
    ts = [np.repeat(theta_mle.reshape(1, dt), nf + 1, axis=0)]
    mesh_grid = [np.concatenate([xc, th], axis=1).tolist() for xc in xs for th in ts]
    mesh_grid = np.array([m for mesh in mesh_grid for m in mesh])

    # Construct obsvar
    obsvar_temp = synth_info.realvar(x_temp)
    obsvar_cand = synth_info.realvar(mesh_grid[:, 0:dx])
    obsvar3D = np.zeros(shape=(nx_ref, nf+1, nf+1)) 
    

    for i in range(nx_ref):
        obsvar3D[i, :, :] = np.diag(np.concatenate([obsvar_temp, np.array([obsvar_cand[i]])]))
    
    n_x = nf + 1
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D)

    for xt_id, x_c in enumerate(xclist):
        x_cand = x_c.reshape(1, dx)
        xt_cand = np.concatenate([x_cand, theta_mle], axis=1)
        
    
        eivar_val[xt_id] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean)


    maxid = np.argmax(eivar_val)
    xnew  = xclist[maxid]

    #xnew = prior_func_x.rnd(1, None).flatten()

    
    xnew_var = synth_info.realvar(xnew)
    for i in range(1):
        y_temp   = synth_info.genobsdata(xnew, xnew_var) 
        des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
        
    #y_temp   = synth_info.genobsdata(xnew, xnew_var) #synth_info.function(xnew, synth_info.true_theta) + np.random.normal(0, np.sqrt(xnew_var), 1) #np.random.normal(0, np.sqrt(synth_info.sigma2), 1)
    #des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
            
    print(des)
                
    return des  

