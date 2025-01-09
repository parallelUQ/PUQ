import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen, twoD, heatmap
import pandas as pd
from PUQ.design import designer
from smt.sampling_methods import LHS
from PUQ.surrogate import emulator
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    multiple_pdfs,
)
import matplotlib.pyplot as plt
from PUQ.surrogatemethods.pcHetGP import update

def test(theta_test, f_test, Xpl, Ypl):
    # ntest x d
    pred = emu.predict(x=cls_func.x, theta=theta_test)
    mu = pred.mean().T 
    
    error = f_test - mu
    for i in range(error.shape[1]):
        plt.hist(error[:, i])
        plt.show()

    S = pred._info['S'] 
    St = np.transpose(S, (2, 0, 1))
    N = St + obsvar3d
    phat = multiple_pdfs(cls_func.real_data, mu, N)
    
    fig, ax = plt.subplots(figsize=(5,5))
    cp = ax.contour(Xpl, Ypl, p_test.reshape(nmesh, nmesh), 20, cmap="RdGy")
    ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,5))
    cp = ax.contour(Xpl, Ypl, phat.reshape(nmesh, nmesh), 20, cmap="RdGy")
    ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    plt.show()
        
args = parse_arguments()

# # # # # 
args.minibatch = 16
args.funcname = 'bimodal'
args.seedmin = 1
args.seedmax = 2
# # # # # 

workers = args.minibatch + 1
n0 = 50
rep0 = 2
nmesh = 50
rho = 1/2
batch = args.minibatch
maxiter = 128
dfl, dfr = [], []
if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):
        
        cls_func = eval(args.funcname)()
        cls_func.realdata(seed=s)

        theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
        test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

        heatmap(cls_func)
    
        # Set a uniform prior
        prior_func = prior_dist(dist="uniform")(
            a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
        )
        
        # Set random stream for initial design
        persis_info = {'rand_stream': np.random.default_rng(s)}
        
        # Initial sample
        sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
        theta0 = sampling(n0)
        theta0 = np.repeat(theta0, rep0, axis=0)
        f0     = np.zeros((cls_func.d, n0*rep0))
        for i in range(0, n0*rep0):
            f0[:, i] = cls_func.sim_f(theta0[i, :], persis_info=persis_info)


        emu = emulator(x=cls_func.x, 
                       theta=theta0, 
                       f=f0,                
                       method="pcHetGP",
                       args={'lower':None, 'upper':None,
                              'noiseControl':{'k_theta_g_bounds': (1, 100), 'g_max': 1e2, 'g_bounds': (1e-6, 1)}, 
                              'init':{}, 
                              'known':{}, 
                               'settings':{"linkThetas": 'joint', "logN": True, "initStrategy": 'residuals', 
                                         "checkHom": True, "penalty": True, "trace": 0, "return.matrices": True, 
                                         "return.hom": False, "factr": 1e9},
                               'pc_settings':{'standardize':True, 'latent':False}})
        
        d = len(cls_func.x)
        obsvar3d = cls_func.obsvar.reshape(1, d, d) 
        
        test(theta_test, f_test, Xpl, Ypl)
        rep_new = 5
        for j in range(0, 5):
            theta_new = np.random.uniform(0, 1, 2)[None, :] #sampling(1)
            print(theta_new)
            update(emu._info, x=cls_func.x, X0new=theta_new, mult=rep_new)
            test(theta_test, f_test, Xpl, Ypl)