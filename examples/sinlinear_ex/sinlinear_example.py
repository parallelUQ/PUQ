#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:57:51 2022

@author: ozgesurer
"""

import seaborn as sns
import pandas as pd
import scipy.stats as sps
from generate_test_data import generate_test_data
import numpy as np
import matplotlib.pyplot as plt
from paractive.design import designer
from paractive.designmethods.utils import parse_arguments, save_output, read_output


def prior_1d(n, thetalimits, seed=None):
    """Generate and return n parameters for the test function."""
    if seed == None:
        pass
    else:
        np.random.seed(seed)
    class prior_uniform:                                                                            
        def rnd(n):
            thlist = []
            for i in range(1):
                thlist.append(sps.uniform.rvs(thetalimits[i][0], thetalimits[i][1]-thetalimits[i][0], size=n))
            return np.array(thlist).T
    thetas = prior_uniform.rnd(n)
    return thetas

class sinlinear:
    def __init__(self):
        self.data_name   = 'sinlinear'
        self.thetalimits = np.array([[-10, 10]])
        self.obsvar = np.array([[1]], dtype='float64')
        self.real_data = np.array([[0]], dtype='float64') 
        self.out = [('f', float)]
        self.p           = 1
        self.d           = 1
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta):
        f = np.sin(theta) + 0.1*theta
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta = H['thetas'][0]
        H_o['f'] = function(theta)

        return H_o, persis_info
    
args        = parse_arguments()
cls_sinlin  = sinlinear()
test_data   = generate_test_data(cls_sinlin)

al_banana = designer(data_cls=cls_sinlin, 
                     method='SEQCAL', 
                     args={'mini_batch': args.minibatch, 
                           'n_init_thetas': 5,
                           'nworkers': args.nworkers,
                           'AL': args.al_func,
                           'seed_n0': args.seed_n0,
                           'prior': prior_1d,
                           'data_test': test_data,
                           'max_evals': 20})


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
    ax.plot(test_data['theta'][:, 0], test_data['p'], color='black')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$p(y|\theta)$')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(test_data['theta'][:, 0], test_data['f'], color='black')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\eta(\theta)$')
    plt.show()    