import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen
from smt.sampling_methods import LHS
import time

args = parse_arguments()

funcs = ['unimodal', 'banana', 'bimodal']

seeds = [1, 4, 28] # 1st submit
seeds = [3, 2, 2]
# unimodal : 5
# banana: 

workers = args.minibatch + 1
n0 = 15
rep0 = 2
nmesh = 50
rho = 1/2
batch = 16
maxiter = 128
workers = batch + 1


if __name__ == "__main__":
    design_start = time.time()
    
    # for s in range(0, 30):
    for ids, fun in enumerate(funcs):
        dictlist = []
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
        sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
        theta0 = sampling(n0)
        theta0 = np.repeat(theta0, rep0, axis=0)
        f0     = np.zeros((cls_func.d, n0*rep0))
        for i in range(0, n0*rep0):
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
                "des_settings":{'is_exploit':True, 'is_explore':True, 'nL':200, 'impute_str': 'update'},
                "trace":0,
            },
        )
        
        dictlist.append({'method': 'ivar', 
                         'theta0': al_ivar._info['theta0'], 
                         'reps0': al_ivar._info['reps0']})

        al_imse = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="simse",
            args={
                "prior": prior_func,
                "data_test": test_data,
                "max_iter": maxiter,
                "nworkers": workers,
                "batch_size": batch,
                "des_init":{'seed':s, 'theta':theta0, 'f':f0},
                "alloc_settings":{'method':'imse', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':True, 'nL':200, 'impute_str': 'update'},
                "trace":0,
            },
        )
        
        dictlist.append({'method': 'imse', 
                         'theta0': al_imse._info['theta0'], 
                         'reps0': al_imse._info['reps0']})
        
        al_var = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="var",
            args={
                "prior": prior_func,
                "data_test": test_data,
                "max_iter": maxiter,
                "nworkers": workers,
                "batch_size": batch,
                "des_init":{'seed':s, 'theta':theta0, 'f':f0},
                "alloc_settings":{'method':'ivar', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':True, 'nL':200, 'impute_str': 'update'},
                "trace":0,
            },
        )
                
        dictlist.append({'method': 'var', 
                         'theta0': al_var._info['theta0'], 
                         'reps0': al_var._info['reps0']})
        
        
        from utilities import twodpaperrev
        twodpaperrev(cls_func=cls_func, 
                  Xpl=Xpl, 
                  Ypl=Ypl, 
                  p_test=p_test, 
                  dictl=dictlist, 
                  thetainit=theta0,
                  name='Figure7_' + fun + 'rev' + str(s) + '.png')
    
    design_end = time.time()
    print("Elapsed time: " + str(round(design_end - design_start, 2)))
