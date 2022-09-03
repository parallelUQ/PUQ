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

    lb1 = thetalimits[0, :][0]
    ub1 = thetalimits[0, :][1]
    lb2 = thetalimits[1, :][0]
    ub2 = thetalimits[1, :][1]
    n1 = 50
    n2 = 50
    xpl = np.linspace(lb1, ub1, n1)
    ypl = np.linspace(lb2, ub2, n2)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    if sh < 2:
        ftest = np.zeros(n1*n2)
    else:
        ftest = np.zeros((2, n1*n2))
    
    ptest = np.zeros(n1*n2)
    thetatest = np.zeros((n1*n2, 2))
    k = 0
 
    for i in range(Xpl.shape[0]):
        for j in range(Xpl.shape[1]): 
            f = function(Xpl[i, j], Ypl[i, j])

            ftest[k] = f
            ptest[k] = compute_likelihood(f.reshape((sh, 1)), np.array([0]), real_data, obsvar, is_cov=False)

            thetatest[k, 0] = Xpl[i, j]
            thetatest[k, 1] = Ypl[i, j]
            k += 1
            
    return {'theta':thetatest, 'p':ptest, 'f':ftest}