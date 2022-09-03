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

    lb1 = thetalimits[0, 0]
    ub1 = thetalimits[0, 1]

    thetatest = np.arange(lb1, ub1, 0.0025)[:, None]
    thetatest = np.append(thetatest, np.array([-6.4, 6.4, -3.6])[:, None], axis=0)
    thetatest = np.sort(thetatest, axis=0)

    
    ftest = np.zeros(thetatest.shape[0])
    ptest = np.zeros(thetatest.shape[0])
    for tt_id, tt in enumerate(thetatest):
        f = function(tt)
        ftest[tt_id] = f
        ptest[tt_id] = compute_likelihood(f.reshape((sh, 1)), np.array([0]), real_data, obsvar, is_cov=False)
    
    
    return {'theta':thetatest, 'p':ptest, 'f':ftest}