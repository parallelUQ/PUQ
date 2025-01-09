from utilities import heatmap
from test_funcs import bimodal, banana, unimodal
from smt.sampling_methods import LHS
import numpy as np
import matplotlib.pyplot as plt

def heatmapestimate(cls_func):

    nmesh = 50
    a = np.arange(nmesh+1)/nmesh
    b = np.arange(nmesh+1)/nmesh
    X, Y = np.meshgrid(a, b)
    Z = np.zeros((nmesh+1, nmesh+1))
    P = np.zeros((nmesh+1, nmesh+1))
    O = np.zeros((nmesh+1, nmesh+1))
    for i in range(nmesh+1):
        for j in range(nmesh+1):
            rrr = np.zeros((100, cls_func.d))
            for r in range(100):
                persis_info = {'rand_stream': np.random.default_rng(r)}
                rrr[r, :] = cls_func.sim_f(np.array([X[i, j], Y[i, j]]), persis_info)
                
            P[i, j] = np.var(rrr[:, 0])
            if cls_func.d > 1:
                O[i, j] = np.var(rrr[:, 1])
                
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, P, cmap='Purples', alpha=0.75)
    cbar = fig.colorbar(cs)
    CS = ax.contour(X, Y, P, colors='black')
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    plt.show() 
    
    if cls_func.d > 1:
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, O, cmap='Purples', alpha=0.75)
        cbar = fig.colorbar(cs)
        CS = ax.contour(X, Y, O, colors='black')
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_xlabel(r"$\theta_1$", fontsize=16)
        ax.set_ylabel(r"$\theta_2$", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        plt.show() 


examples = ['unimodal', 'banana', 'bimodal']
for funcname in examples:
    cls_func = eval(funcname)()
    heatmap(cls_func)
    # heatmapestimate(cls_func)
