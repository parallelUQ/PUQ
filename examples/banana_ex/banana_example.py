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
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output


def prior_banana(n, thetalimits, seed=None):
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

class banana:
    def __init__(self):
        self.data_name   = 'banana'
        self.thetalimits = np.array([[-20, 20], [-10, 5]])
        self.obsvar      = np.array([[10**2, 0], [0, 1]]) 
        self.real_data   = np.array([[1, 3]], dtype='float64')  
        self.out         = [('f', float, (2,))]
        self.p           = 2
        self.d           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2):
        f                = np.array([theta1, theta2 + 0.03*theta1**2])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])

        return H_o, persis_info

args        = parse_arguments()
cls_banana  = banana()
test_data   = generate_test_data(cls_banana)

al_banana = designer(data_cls=cls_banana, 
                     method='SEQCAL', 
                     args={'mini_batch': args.minibatch, 
                           'n_init_thetas': 10,
                           'nworkers': args.nworkers,
                           'AL': args.al_func,
                           'seed_n0': args.seed_n0,
                           'prior': prior_banana,
                           'data_test': test_data,
                           'max_evals': 210})

save_output(al_banana, cls_banana.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = True
if show:
    theta_al = al_banana._info['theta']
    TV       = al_banana._info['TV']
    HD       = al_banana._info['HD']
    
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
    pcm = ax.scatter(test_data['theta'][:, 0], test_data['theta'][:, 1], c=test_data['p'], zorder=1)
    ax.scatter(theta_al[10:, 0], theta_al[10:, 1], c='red', s=5, zorder=2)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    fig.colorbar(pcm, ax=ax)
    plt.show()
    
