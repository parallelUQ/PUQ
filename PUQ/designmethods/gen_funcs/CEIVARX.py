import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat
from smt.sampling_methods import LHS
from numpy.random import rand
import scipy.stats as sps

   
def ceivarx(n, 
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
          x_ref=None,
          theta_ref=None, 
          posttest=None,
          type_init=None,
          synth_info=None,
          theta_mle=None):
    

    x_emu = np.arange(0, 1)[:, None ]
    xuniq = np.unique(x, axis=0)

    # Create a candidate list
    if synth_info.data_name == 'covid19':
        clist = construct_candlist_covid(thetalimits, x_ref, prior_func, prior_func_t)
    else:
        clist = construct_candlist(thetalimits, xuniq, prior_func, prior_func_t )

    nx_ref = x_ref.shape[0]
    dx = x_ref.shape[1]
    nt_ref = theta_ref.shape[0]
    dt = theta_ref.shape[1]
    nf = x.shape[0]

    # Get estimate for real data at theta_mle for reference x
    # nx_ref x (d_x + d_t)
    xt_ref = [np.concatenate([xc.reshape(1, dx), theta_mle], axis=1) for xc in x_ref]
    xt_ref = np.array([m for mesh in xt_ref for m in mesh])

    # 1 x nx_ref
    y_ref = emu.predict(x=x_emu, theta=xt_ref).mean()
    # nx_ref x nf
    f_temp_rep = np.repeat(obs, nx_ref, axis=0)
    # nx_ref x (nf + 1)
    f_field_rep = np.concatenate((f_temp_rep, y_ref.T), axis=1)


    xs = [np.concatenate([x, xc.reshape(1, dx)], axis=0) for xc in x_ref]
    ts = [np.repeat(theta_mle.reshape(1, dt), nf + 1, axis=0)]
    mesh_grid = [np.concatenate([xc, th], axis=1).tolist() for xc in xs for th in ts]
    mesh_grid = np.array([m for mesh in mesh_grid for m in mesh])
    
    n_x = nf + 1
    
    # Construct obsvar
    obsvar3D = np.zeros(shape=(nx_ref, n_x, n_x)) 
    for i in range(nx_ref):
        obsvar3D[i, :, :] = np.diag(np.repeat(synth_info.sigma2, n_x)) 
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D)
    eivar_val = np.zeros(len(clist))
    for xt_id, x_c in enumerate(clist):
        xt_cand = x_c.reshape(1, dx + dt)
        eivar_val[xt_id] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean)


    maxid = np.argmax(eivar_val)
    xnew  = clist[maxid].reshape(1, dx + dt)
    return xnew 


def ceivarxbias(n, 
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
          x_ref=None,
          theta_ref=None, 
          posttest=None,
          emubias=None,
          synth_info=None,
          theta_mle=None,
          unknowncov=None):
    

    x_emu = np.arange(0, 1)[:, None ]
    xuniq = np.unique(x, axis=0)

    clist = construct_candlist(thetalimits, xuniq, prior_func, prior_func_t)

    nx_ref = x_ref.shape[0]
    dx = x_ref.shape[1]
    nt_ref = theta_ref.shape[0]
    dt = theta_ref.shape[1]
    nf = x.shape[0]

    # Get estimate for real data at theta_mle for reference x
    # nx_ref x (d_x + d_t)
    xt_ref = [np.concatenate([xc.reshape(1, dx), theta_mle], axis=1) for xc in x_ref]
    xt_ref = np.array([m for mesh in xt_ref for m in mesh])

    # 1 x nx_ref
    y_ref = emu.predict(x=x_emu, theta=xt_ref).mean()

    # nx_ref x nf
    bias_mean = emubias.predict(x)
    f_temp_rep  = np.repeat(obs-bias_mean, nx_ref, axis=0)
    # nx_ref x (nf + 1)
    f_field_rep = np.concatenate((f_temp_rep, (y_ref).T), axis=1)


    xs = [np.concatenate([x, xc.reshape(1, dx)], axis=0) for xc in x_ref]
    ts = [np.repeat(theta_mle.reshape(1, dt), nf + 1, axis=0)]
    mesh_grid = [np.concatenate([xc, th], axis=1).tolist() for xc in xs for th in ts]
    mesh_grid = np.array([m for mesh in mesh_grid for m in mesh])
    
    n_x = nf + 1
    
    # Construct obsvar
    obsvar3D = np.zeros(shape=(nx_ref, n_x, n_x)) 
    for i in range(nx_ref):
        xnew = np.concatenate((x, x_ref[i].reshape(1, dx)), axis=0)
        if unknowncov:
            bias_var = emubias.predictcov(xnew) 
            obsvar3D[i, :, :] = bias_var
        else:
            obsvar3D[i, :, :] = np.diag(np.repeat(synth_info.sigma2, n_x)) 
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D)
    eivar_val = np.zeros(len(clist))
    for xt_id, x_c in enumerate(clist):
        xt_cand = x_c.reshape(1, dx + dt)
        eivar_val[xt_id] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean)


    maxid = np.argmax(eivar_val)
    xnew  = clist[maxid].reshape(1, dx + dt)
    return xnew 

def construct_candlist_covid(thetalimits, xref, prior_func, prior_func_t ):

    n0 = 50
    xref_sample = np.random.choice(a=221, size=50)[:, None]/221
    n_clist = n0*len(xref_sample)
    t_unif = prior_func_t.rnd(n0, None)
    clist = np.array([np.concatenate([xc, th]) for th in t_unif for xc in xref_sample])
    return clist
 
        
def construct_candlist(thetalimits, xuniq, prior_func, prior_func_t ):

    # Create a candidate list
    n0 = 100
    n_clist  = n0*len(xuniq)
    t_unif = prior_func_t.rnd(n0, None)
    clist1 = np.array([np.concatenate([xc, th]) for th in t_unif for xc in xuniq])
    clist2 = prior_func.rnd(n_clist, None)
    clist = np.concatenate((clist1, clist2), axis=0)

    return clist
        