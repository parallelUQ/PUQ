#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:07:50 2022

@author: ozgesurer
"""
import seaborn as sns
import pandas as pd
import scipy.stats as sps
from generate_test_data import generate_test_data
import numpy as np
import matplotlib.pyplot as plt
from paractive.design import designer
from paractive.designmethods.utils import parse_arguments, save_output

def prior_unidentifiable(n, thetalimits, seed=None):
    """Generate and return n parameters for the test function."""
    if seed == None:
        pass
    else:
        np.random.seed(seed)
    class prior_uniform:                                                                            
        def rnd(n):
            thlist = []
            for i in range(2):
                thlist.append(sps.uniform.rvs(thetalimits[i][0], thetalimits[i][1]-thetalimits[i][0], size=n))
            return np.array(thlist).T
    thetas = prior_uniform.rnd(n)
    return thetas

class unidentifiable:
    def __init__(self):

        self.data_name   = 'unidentifiable'
        self.thetalimits = np.array([[-8, 8], [-8, 8]])
        self.obsvar      = np.array([[1/0.01, 0], [0, 1]]) 
        self.real_data   = np.array([[0, 0]], dtype='float64')  
        self.out     = [('f', float, (2,))]
        self.d           = 2
        self.p           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2):
        f                = np.array([theta1, theta2])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the unidentifiable function
        """
        function         = sim_specs['user']['function']
        H_o              = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']         = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info

args                = parse_arguments()
cls_unidentifiable  = unidentifiable()
test_data           = generate_test_data(cls_unidentifiable)

al_unidentifiable = designer(data_cls=cls_unidentifiable, 
                             method='SEQCAL', 
                             args={'mini_batch': args.minibatch, 
                                   'n_init_thetas': 10,
                                   'nworkers': args.nworkers,
                                   'AL': args.al_func,
                                   'seed_n0': args.seed_n0,
                                   'prior': prior_unidentifiable,
                                   'data_test': test_data,
                                   'max_evals': 210})

save_output(al_unidentifiable, cls_unidentifiable.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = False
if show:
    theta_al = al_unidentifiable._info['theta']
    TV       = al_unidentifiable._info['TV']
    HD       = al_unidentifiable._info['HD']
    
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
