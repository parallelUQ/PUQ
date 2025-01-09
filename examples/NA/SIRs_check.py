#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:11:05 2024

@author: ozgesurer
"""
import numpy as np
import matplotlib.pyplot as plt
from sir_funcs import SIR, SEIRDS

# SIR check
np.random.seed(108)
cls_func = eval('SIR')()
S, I, R = cls_func.simulation(np.array([0.5, 0.5]), S0=1000, I0=10, T=100, repl=1, persis_info=None)
#simulation(self, thetas, S0=990, I0=10, T=150, repl=1, persis_info=None)
print(S.shape)
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(np.mean(S, axis=1), c='g', label='S')
axs.plot(np.mean(I, axis=1), c='y', label='I')
axs.plot(np.mean(R, axis=1), c='b', label='R')
axs.legend()
plt.show()

S, I, R = cls_func.simulation(np.array([0.5, 0.5]), S0=1000, I0=1, T=100, repl=100, persis_info=None)

print(S.shape)
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(S, c='g')
axs.plot(I, c='y')
axs.plot(R, c='b')
plt.show()

# SEIRD check
cls_func = eval('SEIRDS')()
S, E, Ir, Id, R, D = cls_func.simulation(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), repl=1, persis_info=None)

print(S.shape)
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(np.mean(S, axis=1), c='g', label='S')
axs.plot(np.mean(E, axis=1), c='y', label='E')
axs.plot(np.mean(Ir, axis=1), c='b', label='Ir')
axs.plot(np.mean(Id, axis=1), c='r', label='Id')
axs.plot(np.mean(R, axis=1), c='pink', label='R')
axs.plot(np.mean(D, axis=1), c='purple', label='D')
axs.legend(loc = 'upper center', bbox_to_anchor = (1.2, 0.8),
          fancybox = True, shadow = True, ncol = 1)
plt.show()

S, E, Ir, Id, R, D = cls_func.simulation(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), repl=100, persis_info=None)

print(S.shape)
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(S, c='g', label='S')
axs.plot(E, c='y', label='E')
axs.plot(Ir, c='b', label='Ir')
axs.plot(Id, c='r', label='Id')
axs.plot(R, c='pink', label='R')
axs.plot(D, c='purple', label='D')
plt.show()

cls_func.truelims[0][0] = 0
cls_func.truelims[6][0] = 0

S, E, Ir, Id, R, D = cls_func.simulation(np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]), repl=1, persis_info=None)

print(S.shape)
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(S, c='g', label='S')
axs.plot(E, c='y', label='E')
axs.plot(Ir, c='b', label='Ir')
axs.plot(Id, c='r', label='Id')
axs.plot(R, c='pink', label='R')
axs.plot(D, c='purple', label='D')
plt.show()

cls_func = eval('SEIRDS')()
cls_func.truelims[5][0] = 0
cls_func.truelims[6][0] = 0
S, E, Ir, Id, R, D = cls_func.simulation(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0]), repl=100, persis_info=None)

print(S.shape)
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(S, c='g', label='S')
axs.plot(E, c='y', label='E')
axs.plot(Ir, c='b', label='Ir')
axs.plot(Id, c='r', label='Id')
axs.plot(R, c='pink', label='R')
axs.plot(D, c='purple', label='D')
plt.show()

cls_func = eval('SEIRDS')()
# beta=0.3, 
#                    delta=0.3, 
#                    gamma_R=0.08, 
#                    gamma_D=0.12, 
#                    mu=0.7, 
#                    omega=0.01, 
#                    epsilon=0.1,
cls_func.truelims[6][0] = 0.1
cls_func.truelims[0][0] = 0.2
cls_func.truelims[5][0] = 0.01
cls_func.truelims[4][0] = 0.005

S, E, Ir, Id, R, D = cls_func.simulation(np.array([0, 0.5, 0.5, 0.5, 0, 0, 0]), T=1095, S0=100000, E0=1, repl=100, persis_info=None)

print(S.shape)
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(S, c='g', label='S')
axs.plot(E, c='y', label='E')
axs.plot(Ir, c='b', label='Ir')
axs.plot(Id, c='r', label='Id')
axs.plot(R, c='pink', label='R')
axs.plot(D, c='purple', label='D')
plt.show()

from sir_funcs import SEIRDSv2
persis_info = {}
persis_info['rand_stream'] = np.random.default_rng(1)
cls_func = eval('SEIRDSv2')()
S, E, Ir, Id, R, D = cls_func.simulation(cls_func.theta_true, repl=1, persis_info=persis_info)

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.plot(S, c='g', label='S')
axs.plot(E, c='y', label='E')
axs.plot(Ir, c='b', label='Ir')
axs.plot(Id, c='r', label='Id')
axs.plot(R, c='pink', label='R')
axs.plot(D, c='purple', label='D')
plt.show()
