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
from ptest_funcs import sinfunc
from plots_design import gather_data, fitemu, predmle, create_test
from smt.sampling_methods import LHS   

cls_data = sinfunc()
#cls_data.realdata(x=np.array([[0.5], [0.5]]), seed=0)
cls_data.realdata(x=np.array([[0.5]]), seed=0)
args         = parse_arguments()

th_vec = (np.arange(0, 100, 10)/100)[:, None]
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
xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
            
test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh[:, None],      
             'xmesh': xmesh[:, None],
             'p_prior': 1} 

prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

#plt.plot(thetamesh, ptest)
#plt.show()
# # # # # # # # # # # # # # # # # # # # # 
seeds = 10
n0 = 10
for s in range(seeds):
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQEXPDESBIAS', 
                           args={'mini_batch': 1,
                                 'n_init_thetas': n0,
                                 'nworkers': 2, 
                                 'AL': 'ceivar',
                                 'seed_n0': s, 
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': 60,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': True})
    
    xt_eivar = al_unimodal._info['theta']
    f_eivar = al_unimodal._info['f']
    thmle = np.mean(xt_eivar[:, 1])
    xthmle = np.concatenate((xmesh[:, None], np.repeat(thmle, len(xmesh)).reshape((len(xmesh), 1))), axis=1)
    fmle      = np.zeros(len(xthmle))
    for t_id, t in enumerate(xthmle):
        fmle[t_id] = cls_data.function( xthmle[t_id, 0],  xthmle[t_id, 1])
        
    des = al_unimodal._info['des']
    xdes = [e['x'] for e in des]
    nx_ref = len(xdes)
    fdes = np.array([e['feval'][0] for e in des]).reshape(1, nx_ref)
    xu_des, xcount = np.unique(xdes, return_counts=True)
    repeatth = np.repeat(cls_data.true_theta, len(xu_des))
    for label, x_count, y_count in zip(xcount, xu_des, repeatth):
        plt.annotate(label, xy=(x_count, y_count), xytext=(x_count, y_count))
    plt.scatter(xt_eivar[0:n0, 0], xt_eivar[0:n0, 1], marker='*')
    plt.scatter(xt_eivar[:, 0][n0:], xt_eivar[:, 1][n0:], marker='+')
    plt.axhline(y =cls_data.true_theta, color = 'r')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.legend()
    plt.show()
    
    plt.hist(xt_eivar[:, 1][n0:])
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.xlim(0, 1)
    plt.show()

    plt.hist(xt_eivar[:, 0][n0:])
    plt.xlabel(r'x')
    plt.xlim(0, 1)
    plt.show()
    
    #

    xdesu = np.array([x for x in xdes])

    mesh_grid = [np.concatenate((xdesu, np.repeat(t, nx_ref).reshape(nx_ref, 1)), axis=1) for t in thetamesh]
    mesh_grid = np.array([m for mesh in mesh_grid for m in mesh])
    ftest       = np.zeros(len(mesh_grid))
    for t_id, t in enumerate(mesh_grid):
        ftest[t_id] = cls_data.function(mesh_grid[t_id, 0], mesh_grid[t_id, 1])

    ptest = np.zeros(thetamesh.shape[0])
    ftest = ftest.reshape(len(thetamesh), nx_ref)
    for i in range(ftest.shape[0]):
        mean = ftest[i, :] 
        rnd = sps.multivariate_normal(mean=mean, cov=np.diag(np.repeat(cls_data.sigma2, nx_ref)))
        ptest[i] = rnd.pdf(fdes)
    #
    
    phat_eivar, pvar_eivar = fitemu(xt_eivar, f_eivar[:, None], mesh_grid, thetamesh, xdesu, fdes, np.diag(np.repeat(cls_data.sigma2, nx_ref))) 
    y_eivar = predmle(xt_eivar, f_eivar[:, None], xthmle)
    
    print(np.sum((fmle - y_eivar)**2))
    
    print(np.mean(np.abs(phat_eivar - ptest)))
    
    # LHS 
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_lhs   = sampling(60)
    f_lhs    = gather_data(xt_lhs, cls_data)
    phat_lhs, pvar_lhs = fitemu(xt_lhs, f_lhs[:, None], mesh_grid, thetamesh, xdesu, fdes, np.diag(np.repeat(cls_data.sigma2, nx_ref))) 
    y_lhs = predmle(xt_lhs, f_lhs[:, None], xthmle)
    
    print(np.sum((fmle - y_lhs)**2))
    
    print(np.mean(np.abs(phat_lhs - ptest)))
    
    
    #from PUQ.surrogate import emulator
    #x_emu = np.arange(0, 1)[:, None ]
    #emu = emulator(x_emu, 
    #               xtheta, 
    #               fevals[:, None], 
    #               method='PCGPexp')


    #xt_test      = np.concatenate((xmesh[:, None], np.repeat(cls_data.true_theta, len(xmesh))[:, None]), axis=1)
    #true_model  = cls_data.function(xt_test[:, 0], xt_test[:, 1])
    #pred        = emu.predict(x=x_emu, theta=xt_test)
    #mean_pred   = pred.mean()
    #var_pred    = np.sqrt(pred.var())
    #plt.plot(xt_test[:, 0], mean_pred.flatten())
    #plt.scatter(xdes, fdes, color='red')
    #plt.plot(xt_test[:, 0], true_model, color='green')
    #plt.fill_between(xt_test[:, 0], mean_pred.flatten() + 2*var_pred.flatten(), mean_pred.flatten() - 2*var_pred.flatten(), alpha=0.5)
    #plt.show()