import numpy as np
from PUQ.surrogatemethods.PCGPexp import postphi, postphimat, postvarmat
from smt.sampling_methods import LHS

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
        
    theta_acq = []
    n_x, p    = x.shape[0], theta.shape[1]
    obsvar3d  = obsvar.reshape(1, n_x, n_x)

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
        #eivar_val = 0
        #for th_r in thetamesh:
        #    xt_ref = np.concatenate((x, np.repeat(th_r, len(x))[:, None]), axis=1)
        #    eivar_val += postphi(emu._info, x, xt_ref, obs, obsvar, xt_c.reshape(1, 2))
        #print(eivar_val)
        if eivar_val > eivar_max:
            eivar_max = 1*eivar_val
            th_max = 1*xt_c
        
        
    
    print(th_max)
    th_cand = th_max.reshape(1, p)
 
    return th_cand  

def add_new_design(prior_func, emu, x, x_emu, theta_mle, th_mesh, true_fevals, obsvar, synth_info, emubias=None, des=None):

    xclist   = prior_func.rnd(100, None)[:, 0]
    totalvar = np.zeros(len(xclist))
    
    for xcid, xc in enumerate(xclist):
        x_cand    = xc.reshape(1, 1)
        x_field   = np.concatenate((x, x_cand), axis=0)
        xt_mle    = np.concatenate((x_cand, np.array([theta_mle]).reshape(1, 1)), axis=1)
        y_temp    = emu.predict(x=x_emu, theta=xt_mle).mean()
        y_field   = np.concatenate((true_fevals, y_temp), axis=1)
        
        xdesign_vec = np.tile(x_field.flatten(), len(th_mesh))
        theta_ref   = np.concatenate((xdesign_vec[:, None], np.repeat(th_mesh, len(x_field))[:, None]), axis=1)
        
        if emubias == None:
            obsvar_field   = np.diag(np.repeat(np.diag(obsvar)[0], y_field.shape[1]))
        else:
            var_hat = emubias.predict(x=x_emu, theta=x_field.reshape(len(x_field), 1)).var()
            obsvar_field = np.diag(var_hat.reshape(-1))

        
        totalvar[xcid] = postvarmat(emu._info, x_field, theta_ref, y_field, obsvar_field)
    
  
    
    
    xnew      = xclist[np.argmin(totalvar)]
    x         = np.concatenate((x, xnew.reshape(1, 1)), axis=0)
    xt_mle    = np.concatenate((xnew.reshape(1, 1), np.array([theta_mle]).reshape(1, 1)), axis=1)
    y_temp    = synth_info.function(xnew, synth_info.true_theta) + np.random.normal(0, np.sqrt(synth_info.sigma2), 1)
    #print(y_temp)
    #y_temp1    = emu.predict(x=x_emu, theta=xt_mle).mean()

    true_fevals = np.concatenate((true_fevals, y_temp[:, None]), axis=1) 
    
    if emubias == None:     
        obsvar      = np.diag(np.repeat(np.diag(obsvar)[0], y_field.shape[1]))
    else:
        var_hat = emubias.predict(x=x_emu, theta=x.reshape(len(x), 1)).var()
        obsvar = np.diag(var_hat.reshape(-1))        
    
    des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
    print(x)
    return x, obsvar, true_fevals, des
                    

def maxtotvar_exp(n, 
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
        
    theta_acq = []
    n_x, p    = x.shape[0], theta.shape[1]
    obsvar3d  = obsvar.reshape(1, n_x, n_x)

    # Create a candidate list
    n_clist = 100*n
    if type_init == 'LHS':
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
        clist   = prior_func.rnd(n_clist, None)
        

    eivar_max = -np.inf
    th_max = 0
    for xt_c in clist:
        eivar_val = postphimat(emu._info, x, thetatest, obs, obsvar, xt_c.reshape(1, 2))
        if eivar_val > eivar_max:
            eivar_max = 1*eivar_val
            th_max = 1*xt_c
    print(th_max)
    th_cand = th_max.reshape(1, p)
    
    xclist = clist[:, 0]
    #print(xclist)
    eivar_max = np.inf
    x_max = 0
    eitemp = np.zeros(len(xclist))
    for xcid, xc in enumerate(xclist):
        x_field = np.concatenate((x, xc.reshape(1,1)), axis=0)
        theta_mle = np.concatenate((xc.reshape(1,1), np.array([np.pi/5]).reshape(1,1)), axis=1)
        #print(theta_mle)
        y_temp = emu.predict(x=np.arange(0,1)[:, None], theta=theta_mle).mean()
        #print(y_temp)
        #print(obs)
        y_field = np.concatenate((obs, y_temp), axis=1)
        
        xdesign_vec = np.tile(x_field.flatten(), len(thetamesh))
        theta_des   = np.concatenate((xdesign_vec[:, None], np.repeat(thetamesh, len(x_field))[:, None]), axis=1)
        #print(theta_des.shape)
        obsvar_field = np.diag(np.repeat(np.diag(obsvar)[0], y_field.shape[1]))
        eitemp[xcid] = postvarmat(emu._info, x_field, theta_des, y_field, obsvar_field)
        
        #if eitemp < eivar_max:
        #    eivar_max = 1*eitemp
        #    x_max = 1*xc
    import matplotlib.pyplot as plt
    print(xclist[np.argsort(eitemp)])
    plt.plot(xclist[np.argsort(eitemp)])
    plt.show()
        #print(eitemp)
       # print(theta_des)
        #print(obsvar)
        

        #print(np.diag(np.repeat(np.diag(obsvar)[0], y_field.shape[1])))
 
    return th_cand  