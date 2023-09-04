import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred

   
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
    y_ref  = emu.predict(x=x_emu, theta=xt_ref).mean()
    #y_ref_var  = emu.predict(x=x_emu, theta=xt_ref).var()
    
    # nx_ref x nf
    f_temp_rep  = np.repeat(obs, nx_ref, axis=0)
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
          theta_mle=None):
    

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
    # bias_mean = emubias.predict(x).T
    bias_mean = emubias.predict(x_emu, x).mean()
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
        #obsvar3D[i, :, :] = np.diag(np.concatenate([bias_var, np.array([bias_ref_var[i]])])) #np.diag(np.repeat(synth_info.sigma2, n_x))  #np.diag(np.concatenate([bias_var, np.array([bias_ref_var[i]])]))
        obsvar3D[i, :, :] = np.diag(np.repeat(synth_info.sigma2, n_x)) 
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D)
    eivar_val = np.zeros(len(clist))
    for xt_id, x_c in enumerate(clist):
        xt_cand = x_c.reshape(1, dx + dt)
        eivar_val[xt_id] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean)


    maxid = np.argmax(eivar_val)
    xnew  = clist[maxid].reshape(1, dx + dt)
    return xnew 

def construct_candlist(thetalimits, xuniq, prior_func, prior_func_t ):
    type_init = 'CMB'
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
        
    return clist
        
    
def ceivarxfig(n, 
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
          x_mesh=None,
          thetamesh=None, 
          posttest=None,
          type_init=None,
          synth_info=None,
          theta_mle=None):
    

    p = theta.shape[1]
    dt = thetamesh.shape[1]
    dx = x.shape[1]
    type_init = 'CMB'
    x_emu      = np.arange(0, 1)[:, None ]
    xuniq = np.unique(x, axis=0)



    nx_ref = x_mesh.shape[0]
    dx = x_mesh.shape[1]
    nt_ref = thetamesh.shape[0]
    dt = thetamesh.shape[1]
    nf = x.shape[0]

    x_ref     = 1*x_mesh 
    # nt_ref x d_t
    theta_ref = 1*thetamesh

    # Get estimate for real data at theta_mle for reference x
    # nx_ref x (d_x + d_t)
    xt_ref = np.concatenate((x_ref, np.repeat(theta_mle, nx_ref).reshape(nx_ref, dt)), axis=1)
    xt_ref = [np.concatenate([xc.reshape(1, dx), theta_mle], axis=1) for xc in x_ref]
    xt_ref = np.array([m for mesh in xt_ref for m in mesh])

    # 1 x nx_ref
    y_ref  = emu.predict(x=x_emu, theta=xt_ref).mean()
    #y_ref_var  = emu.predict(x=x_emu, theta=xt_ref).var()
    
    # nx_ref x nf
    f_temp_rep  = np.repeat(obs, nx_ref, axis=0)
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
    
    nmesh = 50
    a = np.arange(nmesh)/nmesh
    b = np.arange(nmesh)/nmesh
    X, Y = np.meshgrid(a, b)
    Z = np.zeros((nmesh, nmesh))
    for i in range(nmesh):
        for j in range(nmesh):
            xt_cand = np.array([X[i, j], Y[i, j]]).reshape(1, dx + dt)
            Z[i, j] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean) 
    plt.contourf(X, Y, Z, cmap='Purples')
    plt.hlines(synth_info.true_theta, 0, 1, linestyles='dotted', colors='green')
    for xitem in x:
        plt.vlines(xitem, 0, 1, linestyles='dotted', colors='orange')
    ids = np.where(Z==Z.max())
    xnew = np.array([X[ids[0].flatten(), ids[1].flatten()], Y[ids[0].flatten(), ids[1].flatten()]]).reshape(1, dx + dt)
    plt.scatter(xnew[0,0], xnew[0,1], marker='x', c='red')
    plt.scatter(theta[0:10, 0], theta[0:10, 1], marker='o', c='blue')
    plt.scatter(theta[10:, 0], theta[10:, 1], marker='+', c='black')

    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()
    

    return xnew 
