"""Contains supplemental methods for gen function in persistent_surmise_calib.py."""

import numpy as np
from sklearn.metrics import pairwise_distances
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import compute_postvar, compute_eivar, multiple_pdfs, get_emuvar
import scipy

plotting = False
def rnd(n, 
         x, 
         real_x,
         emu, 
         theta, 
         fevals, 
         obs, 
         obsvar, 
         thetalimits, 
         prior_func, 
         emutype='PC',
         candsize=100, 
         refsize=100, 
         thetatest=None, 
         posttest=None):
        
        theta_acq = prior_func(n, thetalimits, None)
        return theta_acq
        
    
def maxvar(n, 
           x, 
           real_x,
           emu, 
           theta, 
           fevals, 
           obs, 
           obsvar, 
           thetalimits, 
           prior_func, 
           emutype='PC',
           candsize=100, 
           refsize=100, 
           thetatest=None, 
           posttest=None):
    
    # Update emulator for uncompleted jobs.
    idnan            = np.isnan(fevals).any(axis=0).flatten()
    theta_uc         = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean()) 
        
    theta_acq, f_acq = [], []
    d, p             = x.shape[0], theta.shape[1]
    obsvar3d         = obsvar.reshape(1, d, d)
    diags            = np.diag(obsvar[real_x, real_x.T])
    d_real           = real_x.shape[0]
    coef             = (2**d_real)*(np.sqrt(np.pi)**d_real)*np.sqrt(np.prod(diags))
    
    # Create a candidate list.
    clist = prior_func(candsize*n, thetalimits, None)

    # Acquire n thetas.
    for i in range(n):
        emupredict     = emu.predict(x, clist)
        emumean        = emupredict.mean()   
        emuvar, is_cov = get_emuvar(emupredict)
        emumeanT       = emumean.T
        emuvarT        = emuvar.transpose(1, 0, 2)
        var_obsvar1    = emuvarT + obsvar3d 
        var_obsvar2    = emuvarT + 0.5*obsvar3d 
        postmean       = multiple_pdfs(obs, emumeanT[:, real_x.flatten()], var_obsvar1[:, real_x, real_x.T])
        postvar        = compute_postvar(obs, emumeanT[:, real_x.flatten()], 
                                         var_obsvar1[:, real_x, real_x.T], 
                                         var_obsvar2[:, real_x, real_x.T], coef)
        
        acq_func       = postvar.copy()
        idc            = np.argmax(acq_func)
        ctheta         = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
        f_acq.append(postmean[idc])
        clist           = np.delete(clist, idc, 0)
        
        # Kriging believer strategy
        fevalsnew       = emu.predict(x=x, theta=ctheta)
        emu.update(theta=ctheta, f=fevalsnew.mean()) 

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq    

def eivar(n, 
          x, 
          real_x,
          emu, 
          theta, 
          fevals, 
          obs, 
          obsvar, 
          thetalimits, 
          prior_func, 
          emutype='PC',
          candsize=100, 
          refsize=100, 
          thetatest=None, 
          posttest=None):
    
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
    if n == 1:
        n_clist = candsize*n
    else:
        n_clist = candsize*int(n) #100*int(np.ceil(np.log(n)))

    clist   = prior_func(n_clist, thetalimits, None)

    ids = np.random.choice(len(thetatest), refsize, replace=False)
    thetaref = thetatest[ids, :]

    for i in range(n):
        emupredict     = emu.predict(x, thetaref)
        emumean        = emupredict.mean()   
        emumeanT       = emumean.T
        emuvar, is_cov = get_emuvar(emupredict)  
        emuvarT        = emuvar.transpose(1, 0, 2)
        var_obsvar1    = emuvarT + obsvar3d
        

        # Get the n_ref x d x d x n_cand phi matrix
        emuphi4d      = emu.acquisition(x=x, 
                                        theta1=thetaref, 
                                        theta2=clist)
        acq_func = []
        
        # Pass over all the candidates
        for c_id in range(len(clist)):
            posteivar = compute_eivar(var_obsvar1[:, real_x, real_x.T], 
                                      emuphi4d[:, real_x, real_x.T, c_id],
                                      emumeanT[:, real_x.flatten()], 
                                      emuvar[real_x, :, real_x.T], 
                                      obs, 
                                      is_cov)
            acq_func.append(posteivar)

        idc = np.argmin(acq_func)   
        ctheta = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
        clist = np.delete(clist, idc, 0)
        
        # Kriging believer strategy
        fevalsnew = emu.predict(x=x, theta=ctheta)
        emu.update(theta=ctheta, f=fevalsnew.mean()) 

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq    
    
def maxexp(n, 
        x,
        real_x,
        emu,
        theta,
        fevals,
        obs,
        obsvar,
        thetalimits,
        prior_func,
        emutype='PC',
        candsize=100, 
        refsize=100, 
        thetatest=None,
        posttest=None):

    # Update emulator for uncompleted jobs.
    idnan    = np.isnan(fevals).any(axis=0).flatten()
    theta_uc = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean()) 
        
    theta_acq, f_acq = [], []
    d, p    = x.shape[0], theta.shape[1]
    obsvar3d  = obsvar.reshape(1, d, d)
    clist = prior_func(candsize*n, thetalimits, None)

    theta_sd = (theta - thetalimits[:, 0])/(thetalimits[:, 1] - thetalimits[:, 0])
    theta_cand_sd = (clist - thetalimits[:, 0])/(thetalimits[:, 1] - thetalimits[:, 0])

    for j in range(n):    
        
        emupredict     = emu.predict(x, clist)
        emumean        = emupredict.mean()   
        emuvar, is_cov = get_emuvar(emupredict)
        emumeanT       = emumean.T
        emuvarT        = emuvar.transpose(1, 0, 2)
        var_obsvar1    = emuvarT + obsvar3d 
        postmean       = multiple_pdfs(obs, 
                                       emumeanT[:, real_x.flatten()], 
                                       var_obsvar1[:, real_x, real_x.T])
        logpostmean    = np.log(postmean)
        
        
        # Diversity term
        d = pairwise_distances(theta_sd, theta_cand_sd, metric='euclidean')
        min_dist = np.min(d, axis=0)
        
        # Acquisition function
        acq_func = logpostmean + np.log(min_dist) 
        
        # New theta
        idc = np.argmax(acq_func)
        theta_sd = np.concatenate((theta_sd, theta_cand_sd[idc][None, :]), axis=0)
        ctheta = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)    
        f_acq.append(postmean[idc])
        clist = np.delete(clist, idc, 0)
        theta_cand_sd = np.delete(theta_cand_sd, idc, 0)
        
        # Kriging believer strategy
        fevalsnew = emu.predict(x=x, theta=ctheta)
        emu.update(theta=ctheta, f=fevalsnew.mean()) 
        
    theta_acq = np.array(theta_acq).reshape((n, p))
    return theta_acq


def pi(n, 
       x, 
       real_x,
       emu, 
       theta, 
       fevals, 
       obs, 
       obsvar, 
       thetalimits, 
       prior_func, 
       emutype='PC',
       candsize=100, 
       refsize=100, 
       thetatest=None, 
       posttest=None):
    
    # Update emulator for uncompleted jobs.
    idnan            = np.isnan(fevals).any(axis=0).flatten()
    theta_uc         = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean()) 
        
    theta_acq, f_acq = [], []
    d, p             = x.shape[0], theta.shape[1]

    # Create a candidate list.
    clist = prior_func(candsize*n, thetalimits, None)

    error = np.sum(np.abs(obs - fevals.T), axis=1)
    delta = np.abs(obs - fevals.T)[np.argmin(error)]
    pimp = np.zeros(len(clist))

    for cd_id, cl in enumerate(clist):
        if emutype == 'nonPC':
            clall          = np.repeat(cl[None, :], len(x), axis=0)
            xs             = x.flatten()
            clall          = np.concatenate((clall, xs[:, None]), axis=1)
            emupredict     = emu.predict(np.arange(0, 1)[:, None], clall)
            ppmean         = (emupredict.mean()).T
            ppvar          = emupredict.var().flatten()
        else:
            pp = emu.predict(x, cl)
            ppmean = pp.mean()
            ppvar = pp.var()
        for l in range(ppmean.shape[0]):
            diff = obs[0, l] - ppmean[l]
            sumvar = np.diag(obsvar)[l] + ppvar[l]
            
            part1 = scipy.stats.norm.cdf((delta[l] - diff)/np.sqrt(sumvar), 0, 1)
            part2 = scipy.stats.norm.cdf((-delta[l] - diff)/np.sqrt(sumvar), 0, 1)
            pimp[cd_id] += (part1 - part2)

    idc = np.argmax(pimp)
    ctheta         = clist[idc, :].reshape((1, p))
    theta_acq.append(ctheta)

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq  

def ei(n, 
       x, 
       real_x,
       emu, 
       theta, 
       fevals, 
       obs, 
       obsvar, 
       thetalimits, 
       prior_func, 
       emutype='PC',
       candsize=100, 
       refsize=100, 
       thetatest=None, 
       posttest=None):
    
    # Update emulator for uncompleted jobs.
    idnan            = np.isnan(fevals).any(axis=0).flatten()
    theta_uc         = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean()) 
        
    p = theta.shape[1]
    theta_acq, f_acq = [], []
    
    # Create a candidate list.
    clist = prior_func(candsize*n, thetalimits, None)
    
    # Best error
    error = np.sum(np.abs(obs - fevals.T), axis=1)
    delta = np.abs(obs - fevals.T)[np.argmin(error)]
    pimp = np.zeros(len(clist))

    for cd_id, cl in enumerate(clist):
        
        if emutype == 'nonPC':
            clall          = np.repeat(cl[None, :], len(x), axis=0)
            xs             = x.flatten()
            clall          = np.concatenate((clall, xs[:, None]), axis=1)
            emupredict     = emu.predict(np.arange(0, 1)[:, None], clall)
            ppmean         = (emupredict.mean()).T
            ppvar          = emupredict.var().flatten()

        else:
            pp = emu.predict(x, cl)
            ppmean = pp.mean()
            ppvar = pp.var()
            
        p1, i1, r1 = 0, 0, 0
        p2, i2, r2 = 0, 0, 0

        for l in range(ppmean.shape[0]):

            diff = obs[0, l] - ppmean[l]

            sumvar = np.diag(obsvar)[l] + ppvar[l]
            p2 = 1 - scipy.stats.norm.cdf((delta[l] - diff)/np.sqrt(sumvar), 0, 1)
            i2 = (diff - delta[l])
            pdfval = scipy.stats.norm.pdf((delta[l] - diff)/np.sqrt(sumvar), 0, 1)        
            r2 = np.sqrt(sumvar)*pdfval

            p1 = scipy.stats.norm.cdf((-delta[l] - diff)/np.sqrt(sumvar), 0, 1)
            i1 = (-diff - delta[l])
            pdfval = scipy.stats.norm.pdf((-delta[l] - diff)/np.sqrt(sumvar), 0, 1)
            r1 = np.sqrt(sumvar)*pdfval

            pimp[cd_id] += -1*((p1*i1 + r1) + (p2*i2 + r2))


    idc = np.argmax(pimp)
    ctheta         = clist[idc, :].reshape((1, p))
    theta_acq.append(ctheta)
    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq  

def hybrid_pi(n, 
       x, 
       real_x,
       emu, 
       theta, 
       fevals, 
       obs, 
       obsvar, 
       thetalimits, 
       prior_func, 
       emutype='PC',
       candsize=100, 
       refsize=100, 
       thetatest=None, 
       posttest=None):
    
    # Update emulator for uncompleted jobs.
    idnan            = np.isnan(fevals).any(axis=0).flatten()
    theta_uc         = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean()) 
        
    n_x, p = x.shape[0], theta.shape[1]
    theta_acq, f_acq = [], []
    obsvar3d  = obsvar.reshape(1, n_x, n_x)
    # Create a candidate list.
    clist = prior_func(candsize*n, thetalimits, None)
    
    if (len(theta) % 2) == 0:
        error = np.sum(np.abs(obs - fevals.T), axis=1)
        delta = np.abs(obs - fevals.T)[np.argmin(error)]
        pimp = np.zeros(len(clist))
        for cd_id, cl in enumerate(clist):
            if emutype == 'nonPC':
                clall          = np.repeat(cl[None, :], len(x), axis=0)
                xs             = x.flatten()
                clall          = np.concatenate((clall, xs[:, None]), axis=1)
                emupredict     = emu.predict(np.arange(0, 1)[:, None], clall)
                ppmean         = (emupredict.mean()).T
                ppvar          = emupredict.var().flatten()
            else:
                pp = emu.predict(x, cl)
                ppmean = pp.mean()
                ppvar = pp.var()
            for l in range(ppmean.shape[0]):
                diff = obs[0, l] - ppmean[l]
                sumvar = np.diag(obsvar)[l] + ppvar[l]
                
                part1 = scipy.stats.norm.cdf((delta[l] - diff)/np.sqrt(sumvar), 0, 1)
                part2 = scipy.stats.norm.cdf((-delta[l] - diff)/np.sqrt(sumvar), 0, 1)
                pimp[cd_id] += (part1 - part2)
        idc = np.argmax(pimp)
        ctheta         = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
    
        theta_acq = np.array(theta_acq).reshape((n, p))
    else:
        ids = np.random.choice(len(thetatest), refsize, replace=False)
        thetaref = thetatest[ids, :]

        for i in range(n):
            emupredict     = emu.predict(x, thetaref)
            emumean        = emupredict.mean()   
            emumeanT       = emumean.T
            emuvar, is_cov = get_emuvar(emupredict)  
            emuvarT        = emuvar.transpose(1, 0, 2)
            var_obsvar1    = emuvarT + obsvar3d
            

            # Get the n_ref x d x d x n_cand phi matrix
            emuphi4d      = emu.acquisition(x=x, 
                                            theta1=thetaref, 
                                            theta2=clist)
            acq_func = []
            
            # Pass over all the candidates
            for c_id in range(len(clist)):
                posteivar = compute_eivar(var_obsvar1[:, real_x, real_x.T], 
                                          emuphi4d[:, real_x, real_x.T, c_id],
                                          emumeanT[:, real_x.flatten()], 
                                          emuvar[real_x, :, real_x.T], 
                                          obs, 
                                          is_cov)
                acq_func.append(posteivar)

            idc = np.argmin(acq_func)   
            ctheta = clist[idc, :].reshape((1, p))
            theta_acq.append(ctheta)
            clist = np.delete(clist, idc, 0)
            
            # Kriging believer strategy
            fevalsnew = emu.predict(x=x, theta=ctheta)
            emu.update(theta=ctheta, f=fevalsnew.mean()) 

        theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq  

def hybrid_ei(n, 
       x, 
       real_x,
       emu, 
       theta, 
       fevals, 
       obs, 
       obsvar, 
       thetalimits, 
       prior_func, 
       emutype='PC',
       candsize=100, 
       refsize=100, 
       thetatest=None, 
       posttest=None):
    
    # Update emulator for uncompleted jobs.
    idnan            = np.isnan(fevals).any(axis=0).flatten()
    theta_uc         = theta[idnan, :]
    if sum(idnan) > 0:
        fevalshat_uc = emu.predict(x=x, theta=theta_uc)
        emu.update(theta=theta_uc, f=fevalshat_uc.mean()) 
        
    n_x, p = x.shape[0], theta.shape[1]
    theta_acq, f_acq = [], []
    obsvar3d  = obsvar.reshape(1, n_x, n_x)
    # Create a candidate list.
    clist = prior_func(candsize*n, thetalimits, None)
    
    if (len(theta) % 2) == 0:
        # Best error
        error = np.sum(np.abs(obs - fevals.T), axis=1)
        delta = np.abs(obs - fevals.T)[np.argmin(error)]
        pimp = np.zeros(len(clist))
    
        for cd_id, cl in enumerate(clist):
            
            if emutype == 'nonPC':
                clall          = np.repeat(cl[None, :], len(x), axis=0)
                xs             = x.flatten()
                clall          = np.concatenate((clall, xs[:, None]), axis=1)
                emupredict     = emu.predict(np.arange(0, 1)[:, None], clall)
                ppmean         = (emupredict.mean()).T
                ppvar          = emupredict.var().flatten()
    
            else:
                pp = emu.predict(x, cl)
                ppmean = pp.mean()
                ppvar = pp.var()
                
            p1, i1, r1 = 0, 0, 0
            p2, i2, r2 = 0, 0, 0
    
            for l in range(ppmean.shape[0]):
    
                diff = obs[0, l] - ppmean[l]
    
                sumvar = np.diag(obsvar)[l] + ppvar[l]
                p2 = 1 - scipy.stats.norm.cdf((delta[l] - diff)/np.sqrt(sumvar), 0, 1)
                i2 = (diff - delta[l])
                pdfval = scipy.stats.norm.pdf((delta[l] - diff)/np.sqrt(sumvar), 0, 1)        
                r2 = np.sqrt(sumvar)*pdfval
    
                p1 = scipy.stats.norm.cdf((-delta[l] - diff)/np.sqrt(sumvar), 0, 1)
                i1 = (-diff - delta[l])
                pdfval = scipy.stats.norm.pdf((-delta[l] - diff)/np.sqrt(sumvar), 0, 1)
                r1 = np.sqrt(sumvar)*pdfval
    
                pimp[cd_id] += -1*((p1*i1 + r1) + (p2*i2 + r2))
    
    
        idc = np.argmax(pimp)
        ctheta         = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
        theta_acq = np.array(theta_acq).reshape((n, p))
    else:
        ids = np.random.choice(len(thetatest), refsize, replace=False)
        thetaref = thetatest[ids, :]

        for i in range(n):
            emupredict     = emu.predict(x, thetaref)
            emumean        = emupredict.mean()   
            emumeanT       = emumean.T
            emuvar, is_cov = get_emuvar(emupredict)  
            emuvarT        = emuvar.transpose(1, 0, 2)
            var_obsvar1    = emuvarT + obsvar3d
            

            # Get the n_ref x d x d x n_cand phi matrix
            emuphi4d      = emu.acquisition(x=x, 
                                            theta1=thetaref, 
                                            theta2=clist)
            acq_func = []
            
            # Pass over all the candidates
            for c_id in range(len(clist)):
                posteivar = compute_eivar(var_obsvar1[:, real_x, real_x.T], 
                                          emuphi4d[:, real_x, real_x.T, c_id],
                                          emumeanT[:, real_x.flatten()], 
                                          emuvar[real_x, :, real_x.T], 
                                          obs, 
                                          is_cov)
                acq_func.append(posteivar)

            idc = np.argmin(acq_func)   
            ctheta = clist[idc, :].reshape((1, p))
            theta_acq.append(ctheta)
            clist = np.delete(clist, idc, 0)
            
            # Kriging believer strategy
            fevalsnew = emu.predict(x=x, theta=ctheta)
            emu.update(theta=ctheta, f=fevalsnew.mean()) 

        theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq  