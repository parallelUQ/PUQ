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

def prior_unimodal(n, thetalimits, seed=None):
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

class unimodal:
    def __init__(self):
        self.data_name   = 'unimodal'
        self.thetalimits = np.array([[-4, 4], [-4, 4]])
        self.obsvar      = np.array([[4]], dtype='float64')
        self.real_data   = np.array([[-6]], dtype='float64')
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2):
        thetas           = np.array([theta1, theta2]).reshape((1, 2))
        S                = np.array([[1, 0.5], [0.5, 1]])
        f                = (thetas @ S) @ thetas.T
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the unimodal function
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info

args         = parse_arguments()
cls_unimodal = unimodal()
test_data    = generate_test_data(cls_unimodal)

al_unimodal = designer(data_cls=cls_unimodal, 
                       method='SEQCAL', 
                       args={'mini_batch': 1, #args.minibatch, 
                             'n_init_thetas': 10,
                             'nworkers': 2, #args.nworkers,
                             'AL': args.al_func,
                             'seed_n0': 6, #args.seed_n0, #6
                             'prior': prior_unimodal,
                             'data_test': test_data,
                             'max_evals': 60})

save_output(al_unimodal, cls_unimodal.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = True
if show:
    theta_al = al_unimodal._info['theta']
    TV       = al_unimodal._info['TV']
    HD       = al_unimodal._info['HD']
    
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
    ax.scatter(theta_al[10:, 0], theta_al[10:, 1], c='red', zorder=2)
    ax.scatter(theta_al[0:10, 0], theta_al[0:10, 1], c='cyan', zorder=2)
    ax.set_xlabel(r'$\theta_1$', fontsize=16)
    ax.set_ylabel(r'$\theta_2$', fontsize=16)
    #cbar = fig.colorbar(pcm, ax=ax)
    #cbar.ax.tick_params(labelsize=16) 
    ax.tick_params(axis='both', labelsize=16)
    plt.show()
    
    from smt.sampling_methods import LHS
    lb1 = -4
    ub1 = 4
    xlimits = np.array([[lb1, ub1], [lb1, ub1]])
    sampling = LHS(xlimits=xlimits, random_state=2)
    n = 50
    x = sampling(n)
    fig, ax = plt.subplots()
    pcm = ax.scatter(test_data['theta'][:, 0], test_data['theta'][:, 1], c=test_data['p'], zorder=1)
    ax.scatter(x[:, 0], x[:, 1], c='red', zorder=2)
    ax.set_xlabel(r'$\theta_1$', fontsize=16)
    ax.set_ylabel(r'$\theta_2$', fontsize=16)
    #cbar = fig.colorbar(pcm, ax=ax)
    #cbar.ax.tick_params(labelsize=16) 
    ax.tick_params(axis='both', labelsize=16)
    plt.show()
    