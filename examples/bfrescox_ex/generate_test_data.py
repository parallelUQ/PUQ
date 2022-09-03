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
            mean = emumean[:, i] 
    
            if is_cov:
                cov = emuvar[:, i, :] + obsvar
            else:
                cov = np.diag(emuvar[:, i]) + obsvar 

            rnd = sps.multivariate_normal(mean=mean, cov=cov)
            ll[i] = rnd.pdf(obs) 

    return ll

def generate_test_data(cls_synth_init):
    from smt.sampling_methods import LHS
    data_name   = cls_synth_init.data_name
    print('Running: ', data_name)
    function    = cls_synth_init.function
    sh          = cls_synth_init.d
    thetalimits = cls_synth_init.thetalimits
    obsvar      = cls_synth_init.obsvar
    real_data   = cls_synth_init.real_data
    n           = 400
    xlimits     = np.array([[thetalimits[0, :][0], thetalimits[0, :][1]],
                            [thetalimits[1, :][0], thetalimits[1, :][1]],
                            [thetalimits[2, :][0], thetalimits[2, :][1]]])
    sampling    = LHS(xlimits=xlimits, random_state=1)
    x           = sampling(n)
    ftest       = np.zeros((sh, n))       
    ptest       = np.zeros(n)
    thetatest   = np.zeros((n, cls_synth_init.p))
  
    real_x = cls_synth_init.real_x
    real_d = real_x.shape[0]

    for i in range(n):
        parameter   = [x[i, 0], x[i, 1], 0.6798, x[i, 2], 1.0941, 0.2763]
        cls_synth_init.generate_input_file(parameter)
        f           = function()
        ftest[:, i] = f
        ptest[i]    = compute_likelihood(f.reshape((real_d, 1)), 
                                         np.zeros((real_d, 1)), 
                                         real_data, 
                                         obsvar, 
                                         is_cov=False)
        thetatest[i, 0] = x[i, 0]
        thetatest[i, 1] = x[i, 1]
        thetatest[i, 2] = x[i, 2]

                        
    return {'theta':thetatest, 'p':ptest, 'f':ftest}  
            
