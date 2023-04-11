"""Contains supplemental methods for gen function in persistent_surmise_calib.py."""

import numpy as np
from sklearn.metrics import pairwise_distances
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import compute_postvar, compute_eivar, multiple_pdfs, get_emuvar
import scipy

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
    #clist = prior_func(candsize*n, thetalimits, None)
    clist = prior_func(candsize, thetalimits, None)
    # Best error
    error = np.sum(np.abs(obs - fevals.T), axis=1)
    delta = np.abs(obs - fevals.T)[np.argmin(error)]
    
    believer = True
    for i in range(n):
        
        pimp    = np.zeros(len(clist))
        pp      = emu.predict(x, clist)
        ppmean  = pp.mean()
        ppvar   = pp.var()
            
        for l in range(ppmean.shape[0]):
            diff   = obs[0, l] - ppmean[l, :]
            sumvar = np.diag(obsvar)[l] + ppvar[l, :]
            
            p2 = 1 - scipy.stats.norm.cdf((delta[l] - diff)/np.sqrt(sumvar), 0, 1)
            i2 = (diff - delta[l])
            pdfval = scipy.stats.norm.pdf((delta[l] - diff)/np.sqrt(sumvar), 0, 1)        
            r2 = np.sqrt(sumvar)*pdfval
        
            p1 = scipy.stats.norm.cdf((-delta[l] - diff)/np.sqrt(sumvar), 0, 1)
            i1 = (-diff - delta[l])
            pdfval = scipy.stats.norm.pdf((-delta[l] - diff)/np.sqrt(sumvar), 0, 1)
            r1 = np.sqrt(sumvar)*pdfval

            pimp += -1*((p1*i1 + r1) + (p2*i2 + r2))
              
      
        if believer == True:
            idc            = np.argmax(pimp)
            ctheta         = clist[idc, :].reshape((1, p))
            theta_acq.append(ctheta)
            
            # Kriging believer strategy
            fevalsnew = emu.predict(x=x, theta=ctheta)
            emu.update(theta=ctheta, f=fevalsnew.mean()) 
        else:
            idc = np.argsort(pimp)[::-1][:n]
            ctheta         = clist[idc, :].reshape((n, p))
            theta_acq.append(ctheta)
            break
        
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