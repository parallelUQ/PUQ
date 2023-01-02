#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:47:19 2022

@author: ozgesurer
"""

import seaborn as sns
import pandas as pd
import scipy.stats as sps
from generate_test_data import generate_test_data
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output

class gaussian3d:
    def __init__(self):
        self.data_name   = 'gaussian3d'
        self.thetalimits = np.array([[-4, 4], [-4, 4], [-4, 4]])
        self.obsvar      = np.array([[1]], dtype='float64') #np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]) 
        self.real_data   = np.array([[0]], dtype='float64') #np.array([[0, 0, 0]], dtype='float64')  
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 3
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2, theta3):
        a = np.array((theta1, theta2, theta3))[:, None]
        b = np.repeat(0.5*a, 3, axis=1).T
        np.fill_diagonal(b, a)
        f = (a.T@b).flatten()
        f = f @ np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]) 
        return np.sum(f)
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the gaussian3d function
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2])
        
        return H_o, persis_info

args        = parse_arguments()
cls_3d      = gaussian3d()
# # # Create a mesh for test set # # # 
thetalimits = cls_3d.thetalimits

from smt.sampling_methods import LHS

n   = 10000
xlimits   = np.array([[thetalimits[0, :][0], thetalimits[0, :][1]], 
                      [thetalimits[1, :][0], thetalimits[1, :][1]],
                      [thetalimits[2, :][0], thetalimits[2, :][1]]])
sampling  = LHS(xlimits=xlimits)
th         = sampling(n)
    
setattr(cls_3d, 'theta', th)

al_3d_test = designer(data_cls=cls_3d, 
                            method='SEQUNIFORM', 
                            args={'mini_batch': 4, 
                                  'n_init_thetas': 10,
                                  'nworkers': 5,
                                  'max_evals': th.shape[0]})

ftest = al_3d_test._info['f']
thetatest = al_3d_test._info['theta']

ptest = np.zeros(thetatest.shape[0])
for i in range(ftest.shape[0]):
    mean = ftest[i] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_3d.obsvar)
    ptest[i] = rnd.pdf(cls_3d.real_data)
            
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest} 
# # # # # # # # # # # # # # # # # # # # # 

al_3d = designer(data_cls=cls_3d, 
                     method='SEQCAL', 
                     args={'mini_batch': args.minibatch, 
                           'n_init_thetas': 10,
                           'nworkers': args.nworkers,
                           'AL': 'pi', #args.al_func,
                           'seed_n0': args.seed_n0,
                           'prior': 'uniform',
                           'data_test': test_data,
                           'max_evals': 1000,
                           'emutype': 'PC',
                           'candsize': args.candsize,
                           'refsize': 10000}) #args.refsize})

save_output(al_3d, cls_3d.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)


show = True
if show:
    theta_al = al_3d._info['theta']
    f_al = al_3d._info['f']
    TV       = al_3d._info['TV']
    HD       = al_3d._info['HD']
    AE       = al_3d._info['AE']
    time     = al_3d._info['time']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[10:])), TV[10:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()

    plt.scatter(np.arange(len(AE[10:])), AE[10:])
    plt.yscale('log')
    plt.ylabel('AE')
    plt.show()
    
    plt.scatter(np.arange(len(time[12:])), time[12:])
    #plt.yscale('log')
    plt.ylabel('Time')
    plt.show()


