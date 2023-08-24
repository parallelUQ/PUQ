import numpy as np
from PUQ.surrogatemethods.PCGPexp import temp_postphimat, postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from numpy.random import rand
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred

def peivareff(prior_func, prior_func_x, emu, x_emu, theta_mle, x_ref, theta_ref, synth_info, des=None, batchfield=1, batchcounter=1):

    # nf x d_x
    x_temp = np.array([e['x'] for e in des])
    # 1 x nf
    f_temp = np.array([np.mean(e['feval']) for e in des])[None, :]
  
    nx_ref = x_ref.shape[0]
    dx = x_ref.shape[1]
    nt_ref = theta_ref.shape[0]
    dt = theta_ref.shape[1]
    nf = x_temp.shape[0]

    # Create a candidate list
    n_clist = 100
    xclist = prior_func_x.rnd(n_clist, None)
    xclist = np.concatenate((xclist, x_temp))

    # Get estimate for real data at theta_mle for reference x
    # nx_ref x (d_x + d_t)
    xt_ref = [np.concatenate([xc.reshape(1, dx), theta_mle.reshape(1, dt)], axis=1).tolist() for xc in x_ref]
    xt_ref = np.array([m for mesh in xt_ref for m in mesh])

    n_x = nf + 1
        
    theta_mle = theta_mle.reshape(1, dt)
    eivar_val = np.zeros(len(xclist))
    
    # 1 x nx_ref
    y_ref = emu.predict(x=x_emu, theta=xt_ref).mean()
    # nx_ref x nf
    f_temp_rep = np.repeat(f_temp, nx_ref, axis=0)
    # nx_ref x (nf + 1)
    f_field_rep = np.concatenate((f_temp_rep, y_ref.T), axis=1)

    xs = [np.concatenate([x_temp, xc.reshape(1, dx)], axis=0) for xc in x_ref]
    ts = [np.repeat(theta_mle, n_x, axis=0)]
    mesh_grid = [np.concatenate([xc, th], axis=1).tolist() for xc in xs for th in ts]
    mesh_grid = np.array([m for mesh in mesh_grid for m in mesh])

    # Construct obsvar
    obsvar_temp = synth_info.realvar(x_temp)
    obsvar_cand = synth_info.realvar(mesh_grid[:, 0:dx])
    obsvar3D = np.zeros(shape=(nx_ref, n_x, n_x)) 
    for i in range(nx_ref):
        obsvar3D[i, :, :] = np.diag(np.concatenate([obsvar_temp, np.array([obsvar_cand[i]])]))
    
    Smat3D, rVh_1_3d, pred_mean = temp_postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D)

    for xt_id, x_c in enumerate(xclist):
        x_cand = x_c.reshape(1, dx)
        xt_cand = np.concatenate([x_cand, theta_mle], axis=1)
        eivar_val[xt_id] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean)
    
    maxid = np.argmax(eivar_val)
    xnew = xclist[maxid]
    xnew_var = synth_info.realvar(xnew)

    if batchcounter < batchfield:
        # Predict temporary y
        x_new_r = xnew.reshape(1, dx)
        xt_new = np.concatenate([x_new_r, theta_mle], axis=1)
        ynew_temp = emu.predict(x=x_emu, theta=xt_new).mean().flatten()
        des.append({'x': xnew, 'feval':[ynew_temp], 'rep': 1, 'isreal': 'No'})
    else:
        y_new = synth_info.genobsdata(xnew, xnew_var) 
        des.append({'x': xnew, 'feval':[y_new], 'rep': 1, 'isreal': 'Yes'})

    print(des)
    return des  


def peivarefffig(prior_func, prior_func_x, emu, x_emu, theta_mle, x_mesh, th_mesh, synth_info, emubias=None, des=None):

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
    xclist = np.concatenate((xclist, x_temp))

    # nx_ref x d_x
    x_ref     = 1*x_mesh 
    # nt_ref x d_t
    theta_ref = 1*th_mesh

    # Get estimate for real data at theta_mle for reference x
    # nx_ref x (d_x + d_t)
    xt_ref = [np.concatenate([xc.reshape(1, dx), theta_mle.reshape(1, dt)], axis=1).tolist() for xc in x_ref]
    xt_ref = np.array([m for mesh in xt_ref for m in mesh])

    # 1 x nx_ref
    y_ref  = emu.predict(x=x_emu, theta=xt_ref).mean()
    # nx_ref x nf
    f_temp_rep  = np.repeat(f_temp, nx_ref, axis=0)
    # nx_ref x (nf + 1)
    f_field_rep = np.concatenate((f_temp_rep, y_ref.T), axis=1)

    xs = [np.concatenate([x_temp, xc.reshape(1, dx)], axis=0) for xc in x_ref]


    eivar_val = np.zeros(len(xclist))
    theta_mle = theta_mle.reshape(1, dt)
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

    if dx == 1:
        nmesh = 1000
        a = np.arange(nmesh)/nmesh
        Z = np.zeros(nmesh)
        for i in range(nmesh):
            x_cand = np.array([a[i]]).reshape(1, dx)
            xt_cand = np.concatenate([x_cand, theta_mle], axis=1)
            Z[i] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean) 
        plt.plot(a, Z)
        for xitem in x_temp:
            plt.vlines(xitem, np.min(Z), np.max(Z), linestyles='dotted', colors='orange')
    
        
        maxid = np.argmax(Z)
        xnew  = np.array([a[maxid]])
        xnew_var = synth_info.realvar(xnew)
        for i in range(1):
            y_temp   = synth_info.genobsdata(xnew, xnew_var) 
            des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
        plt.scatter(xnew, Z[maxid])
        plt.show()  
        print(des)
    else:
        nmesh = 50
        a = np.arange(nmesh)/nmesh
        b = np.arange(nmesh)/nmesh
        X, Y = np.meshgrid(a, b)
        Z = np.zeros((nmesh, nmesh))
        for i in range(nmesh):
            for j in range(nmesh):
                xt_cand = np.array([X[i, j], Y[i, j]]).reshape(1, dx)
                xt_cand = np.concatenate([xt_cand, theta_mle], axis=1)
                Z[i, j] = postphimat(emu._info, n_x, mesh_grid, f_field_rep, obsvar3D, xt_cand, Smat3D, rVh_1_3d, pred_mean) 
        plt.contourf(X, Y, Z, cmap='Purples')

        for xitem in x_temp:
            plt.scatter(xitem[0], xitem[1], marker='*', c='orange')
        ids = np.where(Z==Z.max())
        xnew = np.array([X[ids[0].flatten(), ids[1].flatten()], Y[ids[0].flatten(), ids[1].flatten()]]).flatten()#.reshape(1, dx )
        xnew_var = synth_info.realvar(xnew)
        for i in range(1):
            y_temp   = synth_info.genobsdata(xnew, xnew_var) 
            des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
            
        plt.scatter(xnew[0], xnew[1], marker='x', c='red')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.show()

                
    return des  

