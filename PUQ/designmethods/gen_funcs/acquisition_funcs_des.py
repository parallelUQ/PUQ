import numpy as np
from PUQ.surrogatemethods.PCGPexp import postphi, postphimat, postvarmat, postpred, postphimat2
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
    #xclist   = np.concatenate((xclist, x_temp.reshape(-1)), axis=0) 

    eivar_val = np.zeros(len(xclist))
    for xid, x_c in enumerate(xclist):
        x_cand    = x_c.reshape(1, 1)
        xt_mle    = np.concatenate((x_cand, np.array([theta_mle]).reshape(1, 1)), axis=1)
        y_temp    = emu.predict(x=x_emu, theta=xt_mle).mean()
        if xid < n_clist:
            x_field   = np.concatenate((x_temp, x_cand), axis=0)
            y_field   = np.concatenate((f_temp, y_temp), axis=1)
            r_field   = r_temp + [1]

        else:
            x_field   = 1*x_temp
            y_field   = 1*f_temp 
            r_field   = 1*r_temp
            for eid, e in enumerate(des):
                if x_cand == e['x']:
                    y_field[0][eid] = (np.sum(e['feval']) + y_temp)/(e['rep'] + 1)
                    r_field[eid] += 1
            
        obsvar_field   = np.diag(np.repeat(synth_info.sigma2, y_field.shape[1]))
        obsvar_field   = obsvar_field/r_field
        xt_c           = np.array([x_c, theta_mle[0]])
        xdesign_vec    = np.tile(x_field.flatten(), len(th_mesh))
        thetatest      = np.concatenate((xdesign_vec[:, None], np.repeat(th_mesh, len(x_field))[:, None]), axis=1)
        eivar_val[xid] = postphimat2(emu._info, x_field, thetatest, y_field, obsvar_field, xt_c.reshape(1, 2))


    minid = np.argmin(eivar_val)
    xnew  = xclist[minid]
    plt.scatter(xclist, eivar_val)
    plt.show()
    if minid >= n_clist:
        for eid, e in enumerate(des):
            if xnew == e['x']:
                e['feval'].append(y_temp)
                e['rep'] += 1
    else:
        y_temp    = synth_info.function(xnew, synth_info.true_theta) + np.random.normal(0, np.sqrt(synth_info.sigma2), 1)
        des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})
        
    print(eivar_val)
                
    return des  

def eivar_new_exp2(prior_func, emu, x_emu, theta_mle, th_mesh, synth_info, emubias=None, des=None):
    
    # Update emulator for uncompleted jobs.
    x_temp = np.array([e['x'] for e in des])[:, None]
    f_temp = np.array([np.mean(e['feval']) for e in des])[None, :]
    r_temp = [e['rep'] for e in des]
    
    # Create a candidate list
    n_clist = 100
    clist   = prior_func.rnd(n_clist, None)
    xclist  = clist[:, 0]
    #xclist  = np.concatenate((xclist, x_temp.reshape(-1)), axis=0) 
 

    x_ref     = np.linspace(synth_info.thetalimits[0][0], synth_info.thetalimits[0][1], 100)
    theta_ref = 1*th_mesh
    #print(x_ref)

    eivar_val = np.zeros(len(xclist))
    for xid, x_c in enumerate(xclist):

        for x_refid, x_r in enumerate(x_ref):
            x_cand    = x_r.reshape(1, 1)
            xt_mle    = np.concatenate((x_cand, np.array([theta_mle]).reshape(1, 1)), axis=1)
            y_temp    = emu.predict(x=x_emu, theta=xt_mle).mean()
    
            #x_field   = 1*x_cand
            #y_field   = 1*y_temp
            x_field   = np.concatenate((x_temp, x_cand), axis=0)
            y_field   = np.concatenate((f_temp, y_temp), axis=1)
        
            for eid, e in enumerate(des):
                if x_c == e['x']:
                    repno = 1*e['rep']
                    break
                else:
                    repno = 1
                    
            obsvar_field  = np.diag(np.repeat(synth_info.sigma2, y_field.shape[1]))/repno
 
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

def add_new_design(prior_func, emu, x_emu, theta_mle, th_mesh, synth_info, emubias=None, des=None):

    x_temp = np.array([e['x'] for e in des])[:, None]
    f_temp = np.array([np.mean(e['feval']) for e in des])[None, :]
    r_temp = [e['rep'] for e in des]
    
    
    xclist   = prior_func.rnd(100, None)[:, 0]
    #print(xclist)
    xclist   = np.concatenate((xclist, x_temp.reshape(-1)), axis=0) 
    totalvar = np.zeros(len(xclist))

    #print(r_temp)
    #print(f_temp)
    for xcid, xc in enumerate(xclist):
        if xcid < 100:
            x_cand    = xc.reshape(1, 1)
            x_field   = np.concatenate((x_temp, x_cand), axis=0)
            xt_mle    = np.concatenate((x_cand, np.array([theta_mle]).reshape(1, 1)), axis=1)
            y_temp    = emu.predict(x=x_emu, theta=xt_mle).mean()
            y_field   = np.concatenate((f_temp, y_temp), axis=1)
            r_field   = r_temp + [1]
        else:
            x_cand    = xc.reshape(1, 1)
            x_field   = 1*x_temp
            xt_mle    = np.concatenate((x_cand, np.array([theta_mle]).reshape(1, 1)), axis=1)
            y_temp    = emu.predict(x=x_emu, theta=xt_mle).mean()
            y_field   = 1*f_temp 
            r_field   = 1*r_temp
            for eid, e in enumerate(des):
                if x_cand == e['x']:
                    y_field[0][eid] = (np.sum(e['feval']) + y_temp)/(e['rep'] + 1)
                    r_field[eid] += 1
        
        
        xdesign_vec = np.tile(x_field.flatten(), len(th_mesh))
        theta_ref   = np.concatenate((xdesign_vec[:, None], np.repeat(th_mesh, len(x_field))[:, None]), axis=1)
        
        if emubias == None:
            obsvar_field   = np.diag(np.repeat(synth_info.sigma2, y_field.shape[1]))
        else:
            var_hat      = emubias.predict(x=x_emu, theta=x_field.reshape(len(x_field), 1)).var()
            obsvar_field = np.diag(var_hat.reshape(-1))
        
        #print(r_field)
        #print(x_field)
        #print(y_field)
        #print(obsvar_field/r_field)
        obsvar_field = obsvar_field/r_field
        totalvar[xcid] = postvarmat(emu._info, x_field, theta_ref, y_field, obsvar_field)
                
    plt.scatter(xclist, totalvar)
    plt.show()
    minid = np.argmax(totalvar)
    #print(totalvar)
    xnew      = xclist[minid]
    print(xnew)
    if minid >= 100:
        for eid, e in enumerate(des):
            if xnew == e['x']:
                e['feval'].append(y_temp)
                e['rep'] += 1
    else:
        y_temp    = synth_info.function(xnew, synth_info.true_theta) + np.random.normal(0, np.sqrt(synth_info.sigma2), 1)
        des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})


    return des 

def observe_design(prior_func, emu, x_emu, theta_mle, th_mesh, synth_info, emubias=None, des=None):

    x_temp = np.array([e['x'] for e in des])[:, None]
    f_temp = np.array([np.mean(e['feval']) for e in des])[None, :]
    r_temp = [e['rep'] for e in des]
    
    
    xclist   = prior_func.rnd(100, None)[:, 0]
    #xclist   = np.concatenate((xclist, x_temp.reshape(-1)), axis=0) 
    totalvar = np.zeros(len(xclist))


    for xcid, xc in enumerate(xclist):
        x_cand    = xc.reshape(1, 1)
        #x_field   = 1*x_cand #np.concatenate((x_temp, x_cand), axis=0)
        x_field   = np.concatenate((x_temp, x_cand), axis=0)
        xt_mle    = np.concatenate((x_cand, np.array([theta_mle]).reshape(1, 1)), axis=1)
        y_temp    = emu.predict(x=x_emu, theta=xt_mle).mean()
        #y_field   = 1*y_temp #np.concatenate((f_temp, y_temp), axis=1)
        y_field   = np.concatenate((f_temp, y_temp), axis=1)
        r_field   = r_temp + [1]

        
        xdesign_vec = np.tile(x_field.flatten(), len(th_mesh))
        theta_ref   = np.concatenate((xdesign_vec[:, None], np.repeat(th_mesh, len(x_field))[:, None]), axis=1)
        
      
        obsvar_field   = np.diag(np.repeat(synth_info.sigma2, y_field.shape[1]))
       

        obsvar_field = obsvar_field/r_field
        print(obsvar_field)
        #print(theta_ref)
        postmean, postvar = postpred(emu._info, x_field, theta_ref, y_field, obsvar_field)
        totalvar[xcid] = np.var(postmean)
        #print(postmean.shape)
    plt.scatter(xclist, totalvar)
    plt.show()
    minid = np.argmax(totalvar)
  
    xnew      = xclist[minid]
    print(xnew)
    if minid >= 100:
        for eid, e in enumerate(des):
            if xnew == e['x']:
                e['feval'].append(y_temp)
                e['rep'] += 1
    else:
        y_temp    = synth_info.function(xnew, synth_info.true_theta) + np.random.normal(0, np.sqrt(synth_info.sigma2), 1)
        des.append({'x': xnew, 'feval':[y_temp], 'rep': 1})


    return des               

