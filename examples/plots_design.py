#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:18:15 2023

@author: ozgesurer
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import  postpred
from smt.sampling_methods import LHS

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
    #plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel(r'$\theta$')
    plt.show()
    
    plt.hist(xt[:, 0][ninit:])
    plt.xlabel(r'x')
    plt.xlim(xlim1, xlim2)
    plt.show()

def plot_des(des, xt, n0, cls_data):
    xdes = np.array([e['x'] for e in des])
    fdes = np.array([e['feval'][0] for e in des]).T
    xu_des, xcount = np.unique(xdes, return_counts=True)
    repeatth = np.repeat(cls_data.true_theta, len(xu_des))
    for label, x_count, y_count in zip(xcount, xu_des, repeatth):
        plt.annotate(label, xy=(x_count, y_count), xytext=(x_count, y_count))
    plt.scatter(xt[0:n0, 0], xt[0:n0, 1], marker='*', color='blue')
    plt.scatter(xt[:, 0][n0:], xt[:, 1][n0:], marker='+', color='red')
    plt.axhline(y =cls_data.true_theta, color = 'black')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()
    
def plot_des_pri(xt, cls_data, ninit, nmax):
    xacq = xt[ninit:nmax, 0:2]
    tacq = xt[ninit:nmax, 2]

    plt.hist(tacq)
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.xlim(0, 1)
    plt.show()
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    plt.show()
    
def plot_LHS(xt, cls_data):
    plt.scatter(cls_data.x, np.repeat(cls_data.true_theta, len(cls_data.x)), marker='o', color='black')
    plt.scatter(xt[:, 0], xt[:, 1], marker='*', color='blue')
    plt.axhline(y = cls_data.true_theta, color = 'green')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()

def plot_post(theta, phat, ptest, phatvar):
    if theta.shape[1] == 1:
        plt.plot(theta.flatten(), phat, c='blue', linestyle='dashed')
        plt.plot(theta.flatten(), ptest, c='black')
        plt.fill_between(theta.flatten(), phat-np.sqrt(phatvar), phat+np.sqrt(phatvar), alpha=0.2)
        plt.show()
    else:
        plt.scatter(theta[:, 0], theta[:, 1], c=phat)
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

    fvec   = np.zeros(len(x_vec))
    for x_id, x in enumerate(x_vec):
        fvec[x_id] = cls_data.function(x, cls_data.true_theta) 
        
    fvec += cls_data.bias(x_vec).flatten()
    plt.plot(x_vec, fvec)
        
    for d_id, d in enumerate(cls_data.des):
        for r in range(d['rep']):
            plt.scatter(d['x'], d['feval'][r], color='black')
    plt.xlabel('x')
    plt.legend()
    plt.show()

    
def create_test(cls_data, isbias=False):
    thetamesh   = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 100)[:, None]
    xmesh = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 100)[:, None]
    
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
            if isbias:
                mean += cls_data.bias(cls_data.x).flatten()
            rnd      = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
            ptest[i] = rnd.pdf(cls_data.real_data)
         
        # plt.plot(thetamesh, ptest)
        # plt.show()
    
        
    
    return thetatest, ftest, ptest, thetamesh, xmesh



def create_test_non(cls_data, is_bias=False):
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

    if is_bias:
        biastrue = cls_data.bias(cls_data.x[:, 0], cls_data.x[:, 1])
        ftest = ftest.reshape(n_t, n_x)
        ptest = np.zeros(n_t)
        for j in range(n_t):
            rnd = sps.multivariate_normal(mean=ftest[j, :] + biastrue, cov=cls_data.obsvar)
            ptest[j] = rnd.pdf(cls_data.real_data)
    else:
        ftest = ftest.reshape(n_t, n_x)
        ptest = np.zeros(n_t)
        for j in range(n_t):
            rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
            ptest[j] = rnd.pdf(cls_data.real_data)
        
    #plt.plot(thetamesh, ptest)
    #plt.show()
    
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
            xt_test[k, :] = np.array([cls_data.x[i, 0], cls_data.x[i, 1], thetamesh[j, 0], thetamesh[j, 1]])
            f[k] = cls_data.function(cls_data.x[i, 0], cls_data.x[i, 1], thetamesh[j, 0], thetamesh[j, 1])
            k += 1
            
    ftest = f.reshape(n_t*n_t, n_x)
    ptest = np.zeros(n_t*n_t)
    for j in range(n_t*n_t):
        rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
        ptest[j] = rnd.pdf(cls_data.real_data)
        
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=ptest)
    plt.show()
    
    return xt_test, ftest, ptest, thetamesh, thetamesh


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

def gather_data_goh(xt, cls_data):
    f    = np.zeros(len(xt))
    for t_id, t in enumerate(xt):
        f[t_id] = cls_data.function(xt[t_id, 0], xt[t_id, 1], xt[t_id, 2], xt[t_id, 3])
    
    return f

def predmle(xt, f, xtmle):
    x_emu      = np.arange(0, 1)[:, None ]
    emu = emulator(x_emu, 
                   xt, 
                   f, 
                   method='PCGPexp')

    yhat = emu.predict(x=x_emu, theta=xtmle).mean()
    
    return yhat

def add_result(method_name, phat, ptest, yhat, ytest, s):
    rep = {}
    rep['method'] = method_name
    rep['Posterior Error'] = np.mean(np.abs(phat - ptest))
    rep['Prediction Error'] = np.mean(np.abs(yhat - ytest))
    
    rep['repno'] = s
    return rep


def samplingdata(typesampling, nmax, cls_data, seed, prior_xt, non=False, goh=False):

    if typesampling == 'LHS':
        sampling = LHS(xlimits=cls_data.thetalimits, random_state=seed)
        xt = sampling(nmax)
    elif typesampling == 'Random':
        xt = prior_xt.rnd(nmax, seed=seed)
    elif typesampling == 'Uniform':
        xuniq = np.unique(cls_data.x, axis=0)
        nf = len(xuniq)
        dt = len(cls_data.true_theta)
        if dt == 1:
            t_unif = sps.uniform.rvs(0, 1, size=int(nmax/nf))[:, None]
        else:
            sampling = LHS(xlimits=cls_data.thetalimits[0:2], random_state=seed)
            t_unif   = sampling(int(nmax/nf))
        mesh_grid = [np.concatenate([xuniq, np.repeat(th, nf).reshape((nf, dt))], axis=1) for th in t_unif]
        xt = np.array([m for mesh in mesh_grid for m in mesh])

 
    if non:
        fevals = gather_data_non(xt, cls_data)
    elif goh:
        fevals = gather_data_goh(xt, cls_data)
    else:
        # plot_LHS(xt, cls_data)
        fevals = gather_data(xt, cls_data)

    return xt, fevals