#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:33:58 2022

@author: ozgesurer
"""
import numpy as np
import scipy.stats as sps

def compute_likelihood(emumean, emuvar, obs, obsvar, is_cov):

    if emumean.shape[0] == 1:
        
        emuvar = emuvar.reshape(emumean.shape)
        ll = sps.norm.pdf(obs-emumean, 0, np.sqrt(obsvar + emuvar))
    else:
        ll = np.zeros(emumean.shape[1])
        for i in range(emumean.shape[1]):
            mean = emumean[:, i] #[emumean[0, i], emumean[1, i]]
    
            if is_cov:
                cov = emuvar[:, i, :] + obsvar
            else:
                cov = np.diag(emuvar[:, i]) + obsvar 

            rnd = sps.multivariate_normal(mean=mean, cov=cov)
            ll[i] = rnd.pdf(obs) #rnd.pdf([obs[0, 0], obs[0, 1]])

    return ll

def generate_test_data(cls_synth_init):

    data_name = cls_synth_init.data_name
    print('Running: ', data_name)
    
    function = cls_synth_init.function
    sh = cls_synth_init.d
    thetalimits = cls_synth_init.thetalimits
    obsvar = cls_synth_init.obsvar
    real_data = cls_synth_init.real_data

    from smt.sampling_methods import LHS
    n       = 10000

    lb1 = thetalimits[0, :][0]
    ub1 = thetalimits[0, :][1]


    xlimits = np.array([[lb1, ub1], [lb1, ub1], [lb1, ub1], [lb1, ub1], [lb1, ub1],
                        [lb1, ub1], [lb1, ub1], [lb1, ub1], [lb1, ub1], [lb1, ub1]])
    sampling = LHS(xlimits=xlimits, random_state=1)
    n = 10000
    x = sampling(n)

    #x = prior10d(n, xlimits, seed=None)

    ftest = np.zeros((10, n))       
    ptest = np.zeros(n)
    thetatest = np.zeros((n, 10))
    k = 0

    for i0 in range(n):

        f = function(x[i0, 0], 
                     x[i0, 1], 
                     x[i0, 2], 
                     x[i0, 3], 
                     x[i0, 4], 
                     x[i0, 5],
                     x[i0, 6], 
                     x[i0, 7], 
                     x[i0, 8], 
                     x[i0, 9])
   

        #print(f)
        ftest[:, k] = f
        ptest[k] = compute_likelihood(f.reshape((sh, 1)), np.zeros((sh, 1)), real_data, obsvar, is_cov=False)

        thetatest[k, 0] = x[i0, 0]
        thetatest[k, 1] = x[i0, 1]
        thetatest[k, 2] = x[i0, 2]
        thetatest[k, 3] = x[i0, 3]
        thetatest[k, 4] = x[i0, 4]
        thetatest[k, 5] = x[i0, 5]
        thetatest[k, 6] = x[i0, 6]
        thetatest[k, 7] = x[i0, 7]
        thetatest[k, 8] = x[i0, 8]
        thetatest[k, 9] = x[i0, 9]                                                     
        k += 1
                        
    return {'theta':thetatest, 'p':ptest, 'f':ftest}   
      
