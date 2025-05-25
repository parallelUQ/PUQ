#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 22:55:05 2025

@author: surero
"""

from sir_funcs import SIR
import numpy as np
import matplotlib.pyplot as plt
cls_func = eval("SIR")()

persis_info = {"rand_stream": np.random.default_rng(100)}

ths = [np.array([0.2, 0.2]), np.array([0.4, 0.4])]

for th in ths:
    nrep = 500
    S, I, R = np.zeros((151, nrep)), np.zeros((151, nrep)), np.zeros((151, nrep))
    for i in range(nrep):
        Si, Ii, Ri = cls_func.simulation(th, S0=1000, I0=10, T=150, repl=1, persis_info=persis_info)
    
        S[:, i] = Si.flatten()
        I[:, i] = Ii.flatten()
        R[:, i] = Ri.flatten()
    
    alp = 0.2
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(S, color="#fdc086", alpha=alp)
    ax.plot(I, color="lightgreen", alpha=alp)
    ax.plot(R, color="#fbb4d9", alpha=alp)
    
    ax.plot(np.mean(S, axis=1), color="orange", linewidth=4, label="Susceptible", linestyle="-")
    ax.plot(np.mean(I, axis=1), color="green", linewidth=4, label="Infected", linestyle="--")
    ax.plot(np.mean(R, axis=1), color="#e377c2", linewidth=4, label="Recovered", linestyle=":" )
    
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Number of individuals", fontsize=12)
    # Place one unified legend below the plots
    fig.legend(
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        fontsize=10
    )
    plt.show()

