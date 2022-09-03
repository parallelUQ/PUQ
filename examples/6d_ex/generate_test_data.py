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
    xlimits = np.array([[thetalimits[0, :][0], thetalimits[0, :][1]],
                        [thetalimits[1, :][0], thetalimits[1, :][1]],
                        [thetalimits[2, :][0], thetalimits[2, :][1]],
                        [thetalimits[3, :][0], thetalimits[3, :][1]],
                        [thetalimits[4, :][0], thetalimits[4, :][1]],
                        [thetalimits[5, :][0], thetalimits[5, :][1]]])
    sampling = LHS(xlimits=xlimits, random_state=1)
    x = sampling(n)
    ftest = np.zeros((sh, n))       
    ptest = np.zeros(n)
    thetatest = np.zeros((n, sh))

    for i in range(n):
        f           = function(x[i, 0], x[i, 1], x[i, 2], x[i, 3], x[i, 4], x[i, 5])
        ftest[:, i] = f
        ptest[i]    = compute_likelihood(f.reshape((sh, 1)), np.zeros((sh, 1)), real_data, obsvar, is_cov=False)
        thetatest[i, 0] = x[i, 0]
        thetatest[i, 1] = x[i, 1]
        thetatest[i, 2] = x[i, 2]
        thetatest[i, 3] = x[i, 3]
        thetatest[i, 4] = x[i, 4]
        thetatest[i, 5] = x[i, 5]            

                        
    return {'theta':thetatest, 'p':ptest, 'f':ftest}       
            
