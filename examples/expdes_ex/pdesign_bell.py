#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:19:03 2023

@author: ozgesurer
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from ptest_funcs import bellcurve

cls_data = bellcurve()
cls_data.realdata(0)
args         = parse_arguments()

th_vec      = (np.arange(0, 100, 10)/100)[:, None]
x_vec = (np.arange(0, 100, 1)/100)[:, None]
fvec = np.zeros((len(th_vec), len(x_vec)))
pvec = np.zeros((len(th_vec)))
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_data.function(x, t)
    plt.plot(x_vec, fvec[t_id, :]) 

for d_id, d in enumerate(cls_data.des):
    for r in range(d['rep']):
        plt.scatter(d['x'], d['feval'][r], color='black')
plt.show()

# # # Create a mesh for test set # # # 
thetamesh = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 30)
xmesh = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 30)
xdesign_vec = np.tile(cls_data.x.flatten(), len(thetamesh))
thetatest   = np.concatenate((xdesign_vec[:, None], np.repeat(thetamesh, len(cls_data.x))[:, None]), axis=1)
ftest       = np.zeros(len(thetatest))
for t_id, t in enumerate(thetatest):
    ftest[t_id] = cls_data.function(thetatest[t_id, 0], thetatest[t_id, 1])

ptest = np.zeros(thetamesh.shape[0])
ftest = ftest.reshape(len(thetamesh), len(cls_data.x))
for i in range(ftest.shape[0]):
    mean = ftest[i, :] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
    ptest[i] = rnd.pdf(cls_data.real_data)
            
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh[:, None],      
             'xmesh': xmesh[:, None],
             'p_prior': 1} 

prior_func      = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
plt.plot(thetamesh, ptest)
plt.show()
# # # # # # # # # # # # # # # # # # # # # 
seeds = 1
for s in range(seeds):
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQEXPDESBIAS', 
                           args={'mini_batch': 1,
                                 'n_init_thetas': 20,
                                 'nworkers': 2, 
                                 'AL': 'eivar_exp',
                                 'seed_n0': s, 
                                 'prior': prior_func,
                                 'data_test': test_data,
                                 'max_evals': 160,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': True})
    
    xth = al_unimodal._info['theta']
    plt.plot(al_unimodal._info['TV'][10:])
    plt.yscale('log')
    plt.show()


    plt.scatter(cls_data.x, np.repeat(cls_data.true_theta, len(cls_data.x)), marker='o', color='black')
    plt.scatter(xth[0:20, 0], xth[0:20, 1], marker='*')
    plt.scatter(xth[:, 0][20:], xth[:, 1][20:], marker='+')
    plt.axhline(y = 0.5, color = 'r')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()

    plt.hist(xth[:, 1][20:])
    plt.axvline(x = 0.5, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.show()

    plt.hist(xth[:, 0][20:])
    plt.xlabel(r'x')
    plt.xlim(0, 1)
    plt.show()