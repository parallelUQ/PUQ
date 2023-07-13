#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:18:15 2023

@author: ozgesurer
"""
import matplotlib.pyplot as plt
import numpy as np
from PUQ.surrogate import emulator
import scipy.stats as sps

def plot_EIVAR(xt, cls_data, ninit):
    
    plt.scatter(cls_data.x, np.repeat(cls_data.true_theta, len(cls_data.x)), marker='o', color='black')
    plt.scatter(xt[0:ninit, 0], xt[0:ninit, 1], marker='*', color='blue')
    plt.scatter(xt[:, 0][ninit:], xt[:, 1][ninit:], marker='+', color='red')
    plt.axhline(y = cls_data.true_theta, color = 'green')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()
    
    plt.hist(xt[:, 1][ninit:])
    plt.axvline(x = cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.show()
    
    plt.hist(xt[:, 0][ninit:])
    plt.xlabel(r'x')
    plt.xlim(0, 1)
    plt.show()
    
def plot_LHS(xt, cls_data):
    plt.scatter(cls_data.x, np.repeat(cls_data.true_theta, len(cls_data.x)), marker='o', color='black')
    plt.scatter(xt[:, 0], xt[:, 1], marker='*', color='blue')
    plt.axhline(y = cls_data.true_theta, color = 'green')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()

def obsdata(cls_data):
    th_vec = [0.3, 0.4, 0.5, 0.6, 0.7]
    x_vec  = (np.arange(0, 100, 1)/100)[:, None]
    fvec   = np.zeros((len(th_vec), len(x_vec)))
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for t_id, t in enumerate(th_vec):
        for x_id, x in enumerate(x_vec):
            fvec[t_id, x_id] = cls_data.function(x, t)
        plt.plot(x_vec, fvec[t_id, :], label=r'$\theta=$' + str(t), color=colors[t_id]) 

    for d_id, d in enumerate(cls_data.des):
        for r in range(d['rep']):
            plt.scatter(d['x'], d['feval'][r], color='black')
    plt.xlabel('x')
    plt.legend()
    plt.show()
    
def create_test(cls_data):
    thetamesh   = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 100)
    xdesign_vec = np.tile(cls_data.x.flatten(), len(thetamesh))
    thetatest   = np.concatenate((xdesign_vec[:, None], np.repeat(thetamesh, len(cls_data.x))[:, None]), axis=1)
    ftest       = np.zeros(len(thetatest))
    for t_id, t in enumerate(thetatest):
        ftest[t_id] = cls_data.function(thetatest[t_id, 0], thetatest[t_id, 1])

    ptest = np.zeros(thetamesh.shape[0])
    ftest = ftest.reshape(len(thetamesh), len(cls_data.x))
    for i in range(ftest.shape[0]):
        mean     = ftest[i, :] 
        rnd      = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
        ptest[i] = rnd.pdf(cls_data.real_data)
         
    plt.plot(thetamesh, ptest)
    plt.show()
    
    return thetatest, ftest, ptest, thetamesh 
    
def fitemu(xt, f, xt_test, thetamesh, cls_data):
    x_emu      = np.arange(0, 1)[:, None ]
    emu = emulator(x_emu, 
                   xt, 
                   f, 
                   method='PCGPexp')
    
    emupredict     = emu.predict(x=x_emu, theta=xt_test)
    emumean        = emupredict.mean()
    emumean = emumean.reshape(len(thetamesh), len(cls_data.x))
    
    posttesthat = np.zeros(len(thetamesh))
    for i in range(emumean.shape[0]):
        mean = emumean[i, :] 
        rnd = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
        posttesthat[i] = rnd.pdf(cls_data.real_data)
        
    return posttesthat
    