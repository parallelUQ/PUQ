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
from ptest_funcs import gohbostos

cls_data = gohbostos()
cls_data.realdata(0)
args         = parse_arguments()

n_t = 20
n_x = 4
n_tot = n_t*n_t*n_x
t1 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], n_t)
t2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], n_t)

T1, T2 = np.meshgrid(t1, t2)
TS = np.vstack([T1.ravel(), T2.ravel()])

XT = np.zeros((n_tot, 4))
f = np.zeros((n_tot))
thetamesh = np.zeros((n_t*n_t, 2))
k = 0
for j in range(n_t*n_t):
    for i in range(n_x):
        XT[k, :] = np.array([cls_data.real_x[i, 0], cls_data.real_x[i, 1], TS[0][j], TS[1][j]])
        f[k] = cls_data.function(cls_data.real_x[i, 0], cls_data.real_x[i, 1], TS[0][j], TS[1][j])
        k += 1
    
    thetamesh[j, :] = np.array([TS[0][j], TS[1][j]])
    
ftest = f.reshape(n_t*n_t, n_x)
ptest = np.zeros(n_t*n_t)

for j in range(n_t*n_t):
    rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
    ptest[j] = rnd.pdf(cls_data.real_data)
    
plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=ptest)
plt.show()

test_data = {'theta': XT, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh,    
             'xmesh': thetamesh,
             'p_prior': 1} 


prior_func      = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 

seeds = 1
ninit = 50
for s in range(seeds):
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQEXPDESBIAS', 
                           args={'mini_batch': 1,
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'eivar_exp',
                                 'seed_n0': s, 
                                 'prior': prior_func,
                                 'data_test': test_data,
                                 'max_evals': 100,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': True})
    
    xth = al_unimodal._info['theta']
    plt.plot(al_unimodal._info['TV'][10:])
    plt.yscale('log')
    plt.show()



    plt.scatter(xth[0:ninit, 0], xth[0:ninit, 1], marker='*')
    plt.scatter(xth[ninit:, 0], xth[ninit:, 1], marker='+')
    #plt.axhline(y = 0.5, color = 'r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    


    plt.scatter(xth[0:ninit, 2], xth[0:ninit, 3], marker='*')
    plt.scatter(xth[ninit:, 2], xth[ninit:, 3], marker='+')
    #plt.axhline(y = 0.5, color = 'r')
    plt.xlabel('theta1')
    plt.ylabel('theta2')
    plt.show()

xs = np.unique(xth[ninit:, 0:2], axis=0)

nx = len(xs)
ftestnew = f.reshape(n_t*n_t, n_x)
ptestnew = np.zeros(n_t*n_t)

for j in range(n_t*n_t):
    m = np.zeros(nx)
    for k in range(nx):
        m[k] = cls_data.function(xs[k][0], xs[k][1], thetamesh[j][0], thetamesh[j][1])
    rnd = sps.multivariate_normal(mean=m, cov=cls_data.obsvar)
    ptest[j] = rnd.pdf(cls_data.real_data)
    
plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=ptest)
plt.show()