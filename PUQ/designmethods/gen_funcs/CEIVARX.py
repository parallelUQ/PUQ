import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred
import scipy.optimize as spo
   
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
          thetamesh=None, 
          posttest=None,
          type_init=None,
          synth_info=None):
    

    p = theta.shape[1]
    dt = thetamesh.shape[1]
    dx = x.shape[1]
    type_init = 'CMB'
    x_emu      = np.arange(0, 1)[:, None ]
    xuniq = np.unique(x, axis=0)

    print(xuniq)
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


    x_mesh = 1* thetamesh
    nx_ref = x_mesh.shape[0]
    dx = x_mesh.shape[1]
    nt_ref = thetamesh.shape[0]
    dt = thetamesh.shape[1]
    nf = x.shape[0]

    theta_mle = find_mle(emu, x, x_emu, obs, dx, dt, thetalimits)
    theta_mle = theta_mle.reshape(1, dt)
    print('mle:', theta_mle)
    
    x_ref     = 1*x_mesh 
    # nt_ref x d_t
    theta_ref = 1*thetamesh

    # Get estimate for real data at theta_mle for reference x
    # nx_ref x (d_x + d_t)
    xt_ref = np.concatenate((x_ref, np.repeat(theta_mle, nx_ref).reshape(nx_ref, dt)), axis=1)

    # 1 x nx_ref
    y_ref  = emu.predict(x=x_emu, theta=xt_ref).mean()
    # nx_ref x nf
    f_temp_rep  = np.repeat(obs, nx_ref, axis=0)
    # nx_ref x (nf + 1)
    f_field_rep = np.concatenate((f_temp_rep, y_ref.T), axis=1)

    xs = [np.concatenate([x, xc.reshape(1, dx)], axis=0) for xc in x_ref]


    eivar_val = np.zeros(len(clist))


    ts = [np.repeat(theta_mle.reshape(1, dt), nf + 1, axis=0)]
    mesh_grid = [np.concatenate([xc, th], axis=1).tolist() for xc in xs for th in ts]
    mesh_grid = np.array([m for mesh in mesh_grid for m in mesh])

    # Construct obsvar
    obsvar3D = np.zeros(shape=(nx_ref, nf+1, nf+1)) 
    for i in range(nx_ref):
        obsvar3D[i, :, :] = np.diag(np.repeat(synth_info.sigma2, nf + 1)) 
    
    n_x = nf + 1
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D)

    for xt_id, x_c in enumerate(clist):
        xt_cand = x_c.reshape(1, dx + dt)
        eivar_val[xt_id] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean)

    maxid = np.argmax(eivar_val)
    xnew  = clist[maxid].reshape(1, dx + dt)
    return xnew 

def obj_mle(parameter, args):
    emu = args[0]
    x = args[1]
    x_emu = args[2]
    obs = args[3]
    xp = np.concatenate((x, np.repeat(parameter, len(x)).reshape(len(x), len(parameter))), axis=1)

    emupred = emu.predict(x=x_emu, theta=xp)
    mu_p    = emupred.mean()
    var_p   = emupred.var()
    
    diff    = (obs.flatten() - mu_p.flatten()).reshape((len(x), 1))
    obj     = 0.5*(diff.T@diff)
    return obj.flatten()

def find_mle(emu, x, x_emu, obs, dx, dt, theta_limits):
    bnd = ()
    theta_init = []
    for i in range(dx, dx + dt):
        bnd += ((theta_limits[i][0], theta_limits[i][1]),)
        theta_init.append((theta_limits[i][0] + theta_limits[i][1])/2)
 
    opval = spo.minimize(obj_mle,
                         theta_init,
                         method='L-BFGS-B',
                         options={'gtol': 0.01},
                         bounds=bnd,
                         args=([emu, x, x_emu, obs]))                

    theta_mle = opval.x
    
    return theta_mle