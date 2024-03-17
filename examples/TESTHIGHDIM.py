#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:03:46 2024

@author: ozgesurer
"""
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from PUQ.surrogate import emulator
from plots_design import create_test_highdim, add_result, samplingdata
from ptest_funcs import highdim2
from PUQ.prior import prior_dist

def plotresult(path, ex_name, w, b, r0, rf, method, n0, nf):

    HDlist = []
    TVlist = []
    timelist = []
    for i in range(r0, rf):
        design_saved = read_output(path, ex_name, method, w, b, i)

        TV       = design_saved._info['TV']
        HD       = design_saved._info['HD']

        TVlist.append(TV[n0:nf])
        HDlist.append(HD[n0:nf])
        
        theta = design_saved._info['theta']
        f = design_saved._info['f']
        
        #print(design_saved._info['thetamle'])
        
    avgTV = np.mean(np.array(TVlist), 0)
    sdTV = np.std(np.array(TVlist), 0)
    avgHD = np.mean(np.array(HDlist), 0)
    sdHD = np.std(np.array(HDlist), 0)

    return avgHD, sdHD, avgTV, sdTV, theta, f, design_saved._info['thetamle']

method = ['ceivarx', 'ceivar', 'lhs', 'maxvar', 'imspe']
method = ['imspe']
path = '/Users/ozgesurer/Desktop/JQT_experiments/highdim_ex/'
rep = 11
n0 = 0
nf = 200
ex = 'x10/'
worker = 2
batch = 1
example_name = 'highdim'
size_x = 10
listeee = []

for mid, m in enumerate(method):  
    print(m)
    for s in np.arange(1, rep):
        s = int(s)
        xr = np.concatenate((np.repeat(0.5, size_x)[None, :], np.repeat(0.5, size_x)[None, :], np.repeat(0.5, size_x)[None, :], np.repeat(0.5, size_x)[None, :]), axis=0)           

        cls_data = highdim2()
        cls_data.realdata(xr, seed=s)
        print(cls_data.sigma2)
  
        prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
        prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:size_x, 0], b=cls_data.thetalimits[0:size_x, 1]) 
        prior_t      = prior_dist(dist='uniform')(a=cls_data.thetalimits[size_x:, 0], b=cls_data.thetalimits[size_x:, 1])
  
        priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}
  
        xt_test, ftest, ptest, thetamesh, xmesh = create_test_highdim(cls_data)
        
        cls_data.true_theta = np.repeat(0.5, 12-size_x)
        thbest = cls_data.true_theta[None, :]
        if size_x == 2:
              ytest = cls_data.function(xmesh[:, 0], xmesh[:, 1], 
                                      cls_data.true_theta[0], 
                                      cls_data.true_theta[1],
                                      cls_data.true_theta[2],
                                      cls_data.true_theta[3],
                                      cls_data.true_theta[4],
                                      cls_data.true_theta[5],
                                      cls_data.true_theta[6],
                                      cls_data.true_theta[7],
                                      cls_data.true_theta[8],
                                      cls_data.true_theta[9]).reshape(1, len(xmesh))                 
        elif size_x == 6:
              ytest = cls_data.function(xmesh[:, 0], xmesh[:, 1], 
                                      xmesh[:, 2], xmesh[:, 3], 
                                      xmesh[:, 4], xmesh[:, 5],
                                      cls_data.true_theta[0], 
                                      cls_data.true_theta[1],
                                      cls_data.true_theta[2],
                                      cls_data.true_theta[3],
                                      cls_data.true_theta[4],
                                      cls_data.true_theta[5]).reshape(1, len(xmesh))       
        elif size_x == 10:
              ytest = cls_data.function(xmesh[:, 0], xmesh[:, 1], 
                                      xmesh[:, 2], xmesh[:, 3], 
                                      xmesh[:, 4], xmesh[:, 5], 
                                      xmesh[:, 6], xmesh[:, 7], 
                                      xmesh[:, 8], xmesh[:, 9],
                                      cls_data.true_theta[0], 
                                      cls_data.true_theta[1]).reshape(1, len(xmesh))
            
        avgPOST, sdPOST, avgPRED, sdPRED, theta, f, thetamle = plotresult(path+ex, example_name, worker, batch, s, s+1, method[mid], n0=n0, nf=nf)
        #print(theta[140:150, :])
        x_emu = np.arange(0, 1)[:, None]
        emu = emulator(x_emu, 
                       theta, 
                       f[:, None], 
                       method='PCGPexp')
            
        print(theta.shape)
        #thbest = thetamle[-1]
     
        xtrue_test = [np.concatenate([xc.reshape(1, size_x), thbest], axis=1) for xc in xmesh]
        xtrue_test = np.array([m for mesh in xtrue_test for m in mesh])
        
        predmean = emu.predict(x=x_emu, theta=xtrue_test).mean()
        #print(predmean)
        print(np.mean(np.abs(ytest - predmean)))
        
        xr = np.concatenate((np.repeat(0.25, size_x)[None, :], np.repeat(0.25, size_x)[None, :], np.repeat(0.25, size_x)[None, :], np.repeat(0.25, size_x)[None, :]), axis=0)      
        th = np.repeat(0.5, 12-size_x)
        thbest = th[None, :]
        xt = [np.concatenate([xc.reshape(1, size_x), thbest], axis=1) for xc in xr]
        xt = np.array([m for mesh in xt for m in mesh])
        print(emu.predict(x=x_emu, theta=xt).mean())
        listeee.append(emu.predict(x=x_emu, theta=xt).mean()[0,0])
        
        