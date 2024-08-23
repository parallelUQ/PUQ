"""Contains supplemental methods for gen function in persistent_surmise_calib.py."""

import numpy as np
from sklearn.metrics import pairwise_distances
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import compute_postvar, compute_eivar, multiple_pdfs, get_emuvar
from smt.sampling_methods import LHS
import scipy


  

def imse(n, 
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
    if n == 1:
        n_clist = 100*n
    else:
        n_clist = 100*int(n) #100*int(np.ceil(np.log(n)))

    if type_init == 'LHS':
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
        clist   = prior_func.rnd(n_clist, None)
        
    for i in range(n):
        # Get the n_ref x d x d x n_cand phi matrix
        emuphi4d      = emu.acquisition(x=x, 
                                        theta1=thetatest, 
                                        theta2=clist)
        acq_func = []
        
        #print(emuphi4d.shape)
        # Pass over all the candidates
        for c_id in range(len(clist)):
            #print(emuphi4d[c_id, :, :, 0])
            posteivar = np.sum(emuphi4d[:, :, :, c_id])
            acq_func.append(posteivar)

        idc = np.argmax(acq_func)   
        ctheta = clist[idc, :].reshape((1, p))
        theta_acq.append(ctheta)
        clist = np.delete(clist, idc, 0)
        
        # Kriging believer strategy
        fevalsnew = emu.predict(x=x, theta=ctheta)
        emu.update(theta=ctheta, f=fevalsnew.mean()) 

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
        thetatest=None,
        posttest=None,
        type_init=None):
    

    n_x, p    = x.shape[0], theta.shape[1]
    # Create a candidate list
    if n == 1:
        n_clist = 100*n
    else:
        n_clist = 100*int(n) #100*int(np.ceil(np.log(n)))

    if type_init == 'LHS':
        sampling = LHS(xlimits=thetalimits)
        clist = sampling(n_clist)
    else:
        clist   = prior_func.rnd(n_clist, None)


    maxpost = np.max(fevals)
    
    for i in range(n):
        emupredict     = emu.predict(np.arange(0, 1)[:, None], clist)
        emumean        = emupredict.mean()   
        emuvar         = emupredict.var() 

        zscore = (emumean - maxpost)/np.sqrt(emuvar)

        cdf_cand = scipy.stats.norm.cdf(zscore)
        pdf_cand = scipy.stats.norm.pdf(zscore)
        ei_cand = (emumean - maxpost)*cdf_cand + np.sqrt(emuvar)*pdf_cand
        acqfun = ei_cand 
    
    #print(acqfun)
    #print(acqfun)
    theta_acq = clist[np.argmax(acqfun), :]

    theta_acq = np.array(theta_acq).reshape((n, p))

    return theta_acq    

def maxvar():
    return
def eivar():
    return
def pi():
    return
def hybrid_ei():
    return
def hybrid_pi():
    return
def rnd():
    return