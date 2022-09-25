#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:07:50 2022

@author: ozgesurer
"""
import seaborn as sns
import pandas as pd
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output

class bimodal:
    def __init__(self):

        self.data_name   = 'bimodal'
        self.thetalimits = np.array([[-6, 6], [-4, 8]])
        self.obsvar      = np.array([[1/np.sqrt(0.2), 0], [0, 1/np.sqrt(0.75)]])
        self.real_data   = np.array([[0, 2]], dtype='float64')
        self.out     = [('f', float, (2,))]
        self.d           = 2
        self.p           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):
        f = np.array([theta2 - theta1**2, theta2 - theta1])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        H_o['f'] = function(H['thetas'][0][0], H['thetas'][0][1])

        return H_o, persis_info

args        = parse_arguments()
cls_bimodal = bimodal()

# # # Create a mesh for test set # # # 
xpl = np.linspace(cls_bimodal.thetalimits[0][0], cls_bimodal.thetalimits[0][1], 50)
ypl = np.linspace(cls_bimodal.thetalimits[1][0], cls_bimodal.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
setattr(cls_bimodal, 'theta', th.T)

al_banana_test = designer(data_cls=cls_bimodal, 
                            method='SEQUNIFORM', 
                            args={'mini_batch': 4, 
                                  'n_init_thetas': 10,
                                  'nworkers': 5,
                                  'max_evals': th.shape[1]})

ftest = al_banana_test._info['f']
thetatest = al_banana_test._info['theta']

ptest = np.zeros(thetatest.shape[0])
for i in range(ftest.shape[0]):
    mean = ftest[i, :] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_bimodal.obsvar)
    ptest[i] = rnd.pdf(cls_bimodal.real_data)
            
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest} 
# # # # # # # # # # # # # # # # # # # # # 

al_bimodal = designer(data_cls=cls_bimodal, 
                      method='SEQCAL', 
                      args={'mini_batch': args.minibatch, 
                            'n_init_thetas': 10,
                            'nworkers': args.nworkers,
                            'AL': args.al_func,
                            'seed_n0': args.seed_n0,
                            'prior': 'uniform',
                            'data_test': test_data,
                            'max_evals': 210})

save_output(al_bimodal, cls_bimodal.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = True
if show:
    theta_al = al_bimodal._info['theta']
    TV       = al_bimodal._info['TV']
    HD       = al_bimodal._info['HD']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[10:])), TV[10:])
    plt.yscale('log')
    plt.ylabel('TV')
    plt.show()
    plt.scatter(np.arange(len(HD[10:])), HD[10:])
    plt.yscale('log')
    plt.ylabel('HD')
    plt.show()
    
    fig, ax = plt.subplots()    
    cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap='RdGy')
    ax.scatter(theta_al[10:, 0], theta_al[10:, 1], c='black', marker='+', zorder=2)
    ax.scatter(theta_al[0:10, 0], theta_al[0:10, 1], zorder=2, marker='o', facecolors='none', edgecolors='blue')
    ax.set_xlabel(r'$\theta_1$', fontsize=16)
    ax.set_ylabel(r'$\theta_2$', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    plt.show()