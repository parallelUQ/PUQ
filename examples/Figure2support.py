import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat
from smt.sampling_methods import LHS
from numpy.random import rand
import scipy.stats as sps
import matplotlib.pyplot as plt
   

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
    plt.contourf(X, Y, Z, cmap='Purples', alpha=1)
    plt.hlines(synth_info.true_theta, 0, 1, linestyles='dotted', linewidth=3, colors='orange')
    for xitem in x:
        plt.vlines(xitem, 0, 1, linestyles='dotted', colors='orange', linewidth=3, zorder=1)
    ids = np.where(Z==Z.max())
    xnew = np.array([X[ids[0].flatten(), ids[1].flatten()], Y[ids[0].flatten(), ids[1].flatten()]]).reshape(1, dx + dt)
    plt.scatter(xnew[0,0], xnew[0,1], marker='x', c='cyan', s=200, zorder=2, linewidth=3)
    plt.scatter(theta[0:10, 0], theta[0:10, 1], marker='*', c='blue', s=50)
    plt.scatter(theta[10:, 0], theta[10:, 1], marker='+', c='red', s=200, linewidth=3)
    plt.text(0.8, 0.05, r'$n_t=$'+ str(theta.shape[0]), fontsize=15)
    plt.xlabel(r'$x$', fontsize=16)
    plt.ylabel(r'$\theta$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
        
    if ((theta.shape[0] == 18) or (theta.shape[0] == 21)):
        plt.savefig("Figure2b_" + str(theta.shape[0]) + ".png", bbox_inches="tight")
    plt.show()

    return xnew 

def ceivarfig(n, 
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
    n_x = x.shape[0]
    
    xuniq = np.unique(x, axis=0)


    xt_ref = np.array([np.concatenate([xc, th]) for th in thetamesh for xc in x])
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, xt_ref, obs, obsvar)

    nmesh = 50
    a = np.arange(nmesh)/nmesh
    b = np.arange(nmesh)/nmesh
    X, Y = np.meshgrid(a, b)
    Z = np.zeros((nmesh, nmesh))
    for i in range(nmesh):
        for j in range(nmesh):
            xt_cand = np.array([X[i, j], Y[i, j]]).reshape(1, dx + dt)
            Z[i, j] = postphimat(emu._info, n_x, xt_ref, obs, obsvar, xt_cand, Smat3D, rVh_1_3d, pred_mean)
    
    plt.contourf(X, Y, Z, cmap='Purples', alpha=1)
    plt.hlines(synth_info.true_theta, 0, 1, linestyles='dotted', linewidth=3, colors='orange')
    for xitem in x:
        plt.vlines(xitem, 0, 1, linestyles='dotted', colors='orange', linewidth=3, zorder=1)
    ids = np.where(Z==Z.max())
    xnew = np.array([X[ids[0].flatten(), ids[1].flatten()], Y[ids[0].flatten(), ids[1].flatten()]]).reshape(1, dx + dt)
    plt.scatter(xnew[0,0], xnew[0,1], marker='x', c='cyan', s=200, zorder=2, linewidth=3)
    plt.scatter(theta[0:10, 0], theta[0:10, 1], marker='*', c='blue', s=50)
    plt.scatter(theta[10:, 0], theta[10:, 1], marker='+', c='red', s=200, linewidth=3)
    plt.text(0.8, 0.05, r'$n_t=$'+ str(theta.shape[0]), fontsize=15)
    plt.xlabel(r'$x$', fontsize=16)
    plt.ylabel(r'$\theta$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if ((theta.shape[0] == 11) or (theta.shape[0] == 16)):
        plt.savefig("Figure2a_" + str(theta.shape[0]) + ".png", bbox_inches="tight")
    plt.show()

    return xnew 


