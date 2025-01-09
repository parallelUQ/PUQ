import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen, twoD, heatmap, twodpaper
import pandas as pd
from PUQ.design import designer
from smt.sampling_methods import LHS
import time

args = parse_arguments()

    
funcs = ['unimodal', 'banana', 'bimodal']
funcs = ['unimodal']
seeds = [15, 0, 9]

grid = 10
rep0 = 2
nmesh = 50
rho = 1/2
batch = 32
maxiter = 192
workers = batch + 1

if __name__ == "__main__":
    design_start = time.time()

    for ids, fun in enumerate(funcs):
        print("Running function: " + fun)
        s = seeds[ids]
        cls_func = eval(fun)()
        cls_func.realdata(seed=None)

        theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
        test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

        # Set a uniform prior
        prior_func = prior_dist(dist="uniform")(
            a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
        )
        
        # Set random stream for initial design
        persis_info = {'rand_stream': np.random.default_rng(s)}

        # Initial sample
        theta1 = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], grid)
        theta2 = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], grid)
        th1, th2 = np.meshgrid(theta1, theta2)
        theta0u = np.vstack([th1.ravel(), th2.ravel()]).T
        theta0 = np.repeat(theta0u, rep0, axis=0)
        f0     = np.zeros((cls_func.d, rep0*(grid**2)))
        for i in range(0, rep0*(grid**2)):
            f0[:, i] = cls_func.sim_f(theta0[i, :], persis_info=persis_info)
        
        al_ivar = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="seivar",
            args={
                "prior": prior_func,
                "data_test": test_data,
                "max_iter": maxiter,
                "nworkers": workers,
                "batch_size": batch,
                "des_init":{'seed':s, 'theta':theta0, 'f':f0},
                "alloc_settings":{'method':'ivar', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':False},
                "trace":0,
            },
        )
        
        twodpaper(cls_func=cls_func, 
                  Xpl=Xpl, 
                  Ypl=Ypl, 
                  p_test=p_test, 
                  theta0=al_ivar._info['theta0'], 
                  reps0=al_ivar._info['reps0'], 
                  name='Figure5_' + fun + '_' + 'ivar' + '.png')

        # al_imse = designer(
        #     data_cls=cls_func,
        #     method="p_sto_bseq",
        #     acquisition="simse",
        #     args={
        #         "prior": prior_func,
        #         "data_test": test_data,
        #         "max_iter": maxiter,
        #         "nworkers": workers,
        #         "batch_size": batch,
        #         "des_init":{'seed':s, 'theta':theta0, 'f':f0},
        #         "alloc_settings":{'method':'imse', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
        #         "pc_settings":{'standardize':True, 'latent':False},
        #         "des_settings":{'is_exploit':True, 'is_explore':False},
        #         "trace":0,
        #     },
        # )
        
        # twodpaper(cls_func=cls_func, 
        #           Xpl=Xpl, 
        #           Ypl=Ypl, 
        #           p_test=p_test, 
        #           theta0=al_imse._info['theta0'], 
        #           reps0=al_imse._info['reps0'], 
        #           name='Figure5_' + fun + '_' + 'imse' + '.png')
        
    design_end = time.time()
    print("Elapsed time: " + str(round(design_end - design_start, 2)))

