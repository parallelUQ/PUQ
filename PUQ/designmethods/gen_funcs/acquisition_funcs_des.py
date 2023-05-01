import numpy as np
from PUQ.surrogatemethods.PCGPexp import postphimat, postpred, postphimat2, postphimat3
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt

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
    n_clist = 100*n
    if type_init == 'LHS':
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
        clist   = prior_func.rnd(n_clist, None)
        
    ###
    xdesign_vec = np.tile(x.flatten(), len(thetamesh))
    thetatest   = np.concatenate((xdesign_vec[:, None], np.repeat(thetamesh, len(x))[:, None]), axis=1)
    ###
    
    eivar_max = -np.inf
    th_max = 0
    for xt_c in clist:
        eivar_val = postphimat(emu._info, x, thetatest, obs, obsvar, xt_c.reshape(1, 2))
        if eivar_val > eivar_max:
            eivar_max = 1*eivar_val
            th_max = 1*xt_c
        
    th_cand = th_max.reshape(1, p)
 
    return th_cand  

def eivar_new_exp(prior_func, emu, x_emu, theta_mle, th_mesh, synth_info, emubias=None, des=None):
    
    # Update emulator for uncompleted jobs.
    x_temp = np.array([e['x'] for e in des])[:, None]
    f_temp = np.array([np.mean(e['feval']) for e in des])[None, :]

    r_temp = [e['rep'] for e in des]
    
    # Create a candidate list
    n_clist = 100
    clist   = prior_func.rnd(n_clist, None)
    xclist  = clist[:, 0]

    x_ref     = np.linspace(synth_info.thetalimits[0][0], synth_info.thetalimits[0][1], 100)
    theta_ref = 1*th_mesh
    #print(x_ref)

    eivar_val = np.zeros(len(xclist))
    for xid, x_c in enumerate(xclist):

        for x_refid, x_r in enumerate(x_ref):
            x_cand    = x_r.reshape(1, 1)
            xt_mle    = np.concatenate((x_cand, np.array([theta_mle]).reshape(1, 1)), axis=1)
            y_temp    = emu.predict(x=x_emu, theta=xt_mle).mean()

            for eid, e in enumerate(des):
                if x_r == e['x']:
                    repno = 1*e['rep']
                    r_field = 1*r_temp
                    r_field[eid] += 1 
                    
                    x_field   = 1*x_temp # np.concatenate((x_temp, x_cand), axis=0)
                    y_field   = 1*f_temp #np.concatenate((f_temp, y_temp), axis=1)
                    y_field[0, eid] = (repno*y_field[0, eid] + y_temp)/(repno + 1)
                    break
                else:
                    repno = 1
                    r_field = r_temp + [repno]
                    x_field   = np.concatenate((x_temp, x_cand), axis=0)
                    y_field   = np.concatenate((f_temp, y_temp), axis=1)
                    #print(x_field.shape)
                    #print(y_field.shape)
  
    
  
            obsvar_field  = np.diag(np.repeat(synth_info.sigma2, y_field.shape[1]))/r_field
            
            #print(obsvar_field)
 
            xt_c           = np.array([x_c, theta_mle[0]])
            xdesign_vec    = np.tile(x_field.flatten(), len(theta_ref))
            thetatest      = np.concatenate((xdesign_vec[:, None], np.repeat(theta_ref, len(x_field))[:, None]), axis=1)
            
            eivar_val[xid] += postphimat2(emu._info, x_field, thetatest, y_field, obsvar_field, xt_c.reshape(1, 2))

    print(eivar_val)
    minid = np.argmin(eivar_val)
    xnew  = xclist[minid]
    plt.scatter(xclist[0:100], eivar_val[0:100])
    plt.show()
    
    
    if minid >= n_clist:
        for eid, e in enumerate(des):
            if xnew == e['x']:
                e['feval'].append(y_temp)
                e['rep'] += 1
    else:
        y_temp    = synth_info.function(xnew, synth_info.true_theta) + np.random.normal(0, np.sqrt(synth_info.sigma2), 1)
        des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
        
    print(des)
                
    return des  

def eivar_new_exp_mat(prior_func, emu, x_emu, theta_mle, th_mesh, synth_info, emubias=None, des=None):
    
    # Update emulator for uncompleted jobs.
    x_temp = np.array([e['x'] for e in des])[:, None]
    f_temp = np.array([np.mean(e['feval']) for e in des])[None, :]

    r_temp = [e['rep'] for e in des]
    
    # Create a candidate list
    n_clist = 100
    clist   = prior_func.rnd(n_clist, None)
    xclist  = clist[:, 0]

    nx_ref    = 10
    x_ref     = np.linspace(synth_info.thetalimits[0][0], synth_info.thetalimits[0][1], nx_ref)
    theta_ref = 1*th_mesh

    xt_ref = np.concatenate((x_ref[:, None], np.repeat(theta_mle, nx_ref)[:, None]), axis=1)
    y_ref  = emu.predict(x=x_emu, theta=xt_ref).mean()

    #print(y_ref)
    f_temp_rep  = np.repeat(f_temp, nx_ref, axis=0)
    x_temp_rep  = np.tile(x_temp.T, [nx_ref, 1])
    
    # (d_x + d_ref) x (nx_ref)
    x_field_rep = np.concatenate((x_temp_rep, x_ref[:, None]), axis=1)
    f_field_rep = np.concatenate((f_temp_rep, y_ref.T), axis=1)
    

    # (d_x + d_ref + d_t) x (nx_ref x nt_ref)
    xflat = np.repeat(x_field_rep, theta_ref.shape[0], axis=0) #
    xflat = xflat.flatten()[:, None]
    
    tr = np.repeat(theta_ref, x_field_rep.shape[1])
    tr = np.tile(tr, [nx_ref, 1]).flatten()[:, None]
    xf_thmesh = np.concatenate((xflat, tr), axis=1)
    
    #print(xf_thmesh)
    #print(f_field_rep)
    #print(np.repeat(f_field_rep, th_mesh.shape[0], axis=0))

    
    obsvar_field  = np.diag(np.repeat(synth_info.sigma2, f_field_rep.shape[1]))
    
    eivar_val = np.zeros(len(xclist))
    eivar_val1 = np.zeros(len(xclist))
    for xid, x_c in enumerate(xclist):
        xt_c           = np.array([x_c, theta_mle[0]])
        eivar_val1[xid] = postphimat3(emu._info, xf_thmesh, np.repeat(f_field_rep, th_mesh.shape[0], axis=0), obsvar_field, xt_c.reshape(1, 2))
        
        for x_refid, x_r in enumerate(x_ref):
            x_cand    = x_r.reshape(1, 1)
            y_temp    = y_ref[0, x_refid].reshape(1, 1)


            x_field   = np.concatenate((x_temp, x_cand), axis=0)
            y_field   = np.concatenate((f_temp, y_temp), axis=1)
    
            obsvar_field  = np.diag(np.repeat(synth_info.sigma2, y_field.shape[1]))
            

            xt_c           = np.array([x_c, theta_mle[0]])
            xdesign_vec    = np.tile(x_field.flatten(), len(theta_ref))
            thetatest      = np.concatenate((xdesign_vec[:, None], np.repeat(theta_ref, len(x_field))[:, None]), axis=1)
         
            if xid == 0:
                print(x_field.shape[0])
            eivar_val[xid] += postphimat2(emu._info, x_field, thetatest, y_field, obsvar_field, xt_c.reshape(1, 2))
    #print('thetest')
    #print(thetatest)
    print(np.round(eivar_val-eivar_val1, 1))

    minid = np.argmin(eivar_val)
    xnew  = xclist[minid]
    plt.scatter(xclist[0:100], eivar_val[0:100])
    plt.show()
    
    
    if minid >= n_clist:
        for eid, e in enumerate(des):
            if xnew == e['x']:
                e['feval'].append(y_temp)
                e['rep'] += 1
    else:
        y_temp    = synth_info.function(xnew, synth_info.true_theta) + np.random.normal(0, np.sqrt(synth_info.sigma2), 1)
        des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
        
    print(des)
                
    return des  
