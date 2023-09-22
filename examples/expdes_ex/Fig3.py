import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, obsdata, create_test, add_result, samplingdata
from ptest_funcs import sinfunc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3

args = parse_arguments()

s = 1

cls_data = sinfunc()
dt = len(cls_data.true_theta)
cls_data.realdata(x=np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None], seed=s)
# Observe
#obsdata(cls_data)
        
prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

# # # Create a mesh for test set # # # 
xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
nmesh = len(xmesh)
cls_data_y = sinfunc()
cls_data_y.realdata(x=xmesh, seed=s)
ytest = cls_data_y.real_data

test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'y': ytest,
             'th': thetamesh,    
             'xmesh': xmesh,
             'p_prior': 1} 
# # # # # # # # # # # # # # # # # # # # # 

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

    #xt_ref = [np.concatenate([xc.reshape(1, dx), theta_mle], axis=1) for xc in x_ref]
    #xt_ref = np.array([m for mesh in xt_ref for m in mesh])
    #print(xt_ref)
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

from PUQ.surrogate import emulator
from PUQ.surrogatemethods.PCGPexp import  postpred
x_emu = np.arange(0, 1)[:, None ]

ninit = 10
nmax = 30
xt  = prior_xt.rnd(ninit, s) 
f = cls_data.function(xt[:, 0], xt[:, 1])

for i in range(nmax-ninit):
    emu = emulator(x_emu, 
                   xt, 
                   f[None, :], 
                   method='PCGPexp')
    
    xnew = ceivarxfig(1, 
              cls_data.x, 
              cls_data.x,
              emu, 
              xt, 
              f[None, :], 
              cls_data.real_data, 
              cls_data.obsvar, 
              cls_data.thetalimits, 
              prior_xt,
              prior_t,
              thetatest=None, 
              x_mesh=xmesh,
              thetamesh=thetamesh, 
              posttest=ptest,
              type_init=None,
              synth_info=cls_data,
              theta_mle=np.array([0.62]))
    
    xt = np.concatenate((xt, xnew), axis=0)
    f = cls_data.function(xt[:, 0], xt[:, 1])

    