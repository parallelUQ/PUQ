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

def plot_EIVAR(xt, cls_data, ninit, xlim1=0, xlim2=1):
    
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
    plt.xlim(xlim1, xlim2)
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

def obsdata2(cls_data):
    th_vec = [0.3, 0.4, 0.5, 0.6, 0.7]
    x_vec  = (np.arange(-300, 300, 1)/100)[:, None]
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
    xmesh = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 100)
    
    #if (cls_data.x).all() != None:
    if cls_data.nodata:
        thetatest, ftest, ptest = None, None, None
    else:
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
    
        
    
    return thetatest, ftest, ptest, thetamesh, xmesh



def create_test_non(cls_data):
    n_t = 100
    n_x = cls_data.x.shape[0]
    n_tot = n_t*n_x
    thetamesh = np.linspace(cls_data.thetalimits[2][0], cls_data.thetalimits[2][1], n_t)

    xt_test = np.zeros((n_tot, 3))
    ftest = np.zeros(n_tot)
    k = 0
    for j in range(n_t):
        for i in range(n_x):
            xt_test[k, :] = np.array([cls_data.x[i, 0], cls_data.x[i, 1], thetamesh[j]])
            ftest[k] = cls_data.function(cls_data.x[i, 0], cls_data.x[i, 1], thetamesh[j])
            k += 1

    ftest = ftest.reshape(n_t, n_x)
    ptest = np.zeros(n_t)
    for j in range(n_t):
        rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
        ptest[j] = rnd.pdf(cls_data.real_data)
        
    plt.scatter(thetamesh, ptest)
    plt.show()
    
    x1 = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 20)
    x2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 20)
    X1, X2 = np.meshgrid(x1, x2)
    xmesh = np.vstack([X1.ravel(), X2.ravel()]).T
    
    return xt_test, ftest, ptest, thetamesh[:, None], xmesh

def create_test_goh(cls_data):
    n_t = 20
    n_x = len(cls_data.x)
    n_tot = n_t*n_t*n_x
    t1 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], n_t)
    t2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], n_t)
    T1, T2 = np.meshgrid(t1, t2)
    TS = np.vstack([T1.ravel(), T2.ravel()])
    thetamesh = TS.T
    
    xt_test = np.zeros((n_tot, 4))
    f = np.zeros((n_tot))
    k = 0
    for j in range(n_t*n_t):
        for i in range(n_x):
            xt_test[k, :] = np.array([cls_data.real_x[i, 0], cls_data.real_x[i, 1], thetamesh[j, 0], thetamesh[j, 1]])
            f[k] = cls_data.function(cls_data.real_x[i, 0], cls_data.real_x[i, 1], thetamesh[j, 0], thetamesh[j, 1])
            k += 1
            
    ftest = f.reshape(n_t*n_t, n_x)
    ptest = np.zeros(n_t*n_t)
    for j in range(n_t*n_t):
        rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
        ptest[j] = rnd.pdf(cls_data.real_data)
        
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=ptest)
    plt.show()
    
    return xt_test, ftest, ptest, thetamesh, thetamesh

def fitemu(xt, f, xt_test, thetamesh, x, obs, obsvar):
    x_emu      = np.arange(0, 1)[:, None ]
    emu = emulator(x_emu, 
                   xt, 
                   f, 
                   method='PCGPexp')
    from PUQ.surrogatemethods.PCGPexp import  postpred
    pmeanhat, pvarhat = postpred(emu._info, x, xt_test, obs, obsvar)
    

    #emupredict     = emu.predict(x=x_emu, theta=xt_test)
    #emumean        = emupredict.mean()
    #emumean = emumean.reshape(len(thetamesh), len(cls_data.x))
    
    #posttesthat_c = np.zeros(len(thetamesh))
    #for i in range(emumean.shape[0]):
    #    mean = emumean[i, :] 
    #    rnd = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
    #    posttesthat_c[i] = rnd.pdf(cls_data.real_data)

    #print(np.round(posttesthat_c - posttesthat, 4))
    return pmeanhat, pvarhat

def gather_data(xt, cls_data):
    f    = np.zeros(len(xt))
    for t_id, t in enumerate(xt):
        f[t_id] = cls_data.function(xt[t_id, 0], xt[t_id, 1])
    
    return f

def gather_data_non(xt, cls_data):
    f    = np.zeros(len(xt))
    for t_id, t in enumerate(xt):
        f[t_id] = cls_data.function(xt[t_id, 0], xt[t_id, 1], xt[t_id, 2])
    
    return f

def predmle(xt, f, xtmle):
    x_emu      = np.arange(0, 1)[:, None ]
    emu = emulator(x_emu, 
                   xt, 
                   f, 
                   method='PCGPexp')

    yhat = emu.predict(x=x_emu, theta=xtmle).mean()
    
    return yhat