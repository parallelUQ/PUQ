import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal, branin
from utilities import test_data_gen, twoD, heatmap, twodpaper
import pandas as pd
from PUQ.design import designer
from smt.sampling_methods import LHS

args = parse_arguments()

# # # # # 
args.minibatch = 16
args.funcname = 'branin'
args.seedmin = 4
args.seedmax = 5
# # # # # 

workers = args.minibatch + 1
n0 = 15
rep0 = 2
nmesh = 50
rho = 1/2
batch = args.minibatch
maxiter = 128
dfl, dfr = [], []
if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):
        
        cls_func = eval(args.funcname)()
        cls_func.realdata(seed=None)

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
        # theta0 = np.array([[0, 0], [0.5, 0], [1, 0], 
        #                     [0, 0.5], [0.5, 0.5], [1, 0.5],
        #                     [0, 1], [0.5, 1], [1, 1]])
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
                "des_settings":{'is_exploit':False, 'is_explore':True, 'nL':200, 'impute_str':'KB'}
            },
        )
        
        save_output(al_ivar, cls_func.data_name, 'ivar', workers, batch, s)
        twodpaper(cls_func=cls_func, 
                  Xpl=Xpl, 
                  Ypl=Ypl, 
                  p_test=p_test, 
                  theta0=al_ivar._info['theta0'], 
                  reps0=al_ivar._info['reps0'], 
                  thetainit=theta0,
                  name='branin' + "KB" + '_' + 'ivar' + '.png')
        
        #twoD(al_ivar, Xpl, Ypl, p_test, nmesh)
        
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
                "des_settings":{'is_exploit':False, 'is_explore':True, 'nL':200, 'impute_str':'CL'}
            },
        )
        
        save_output(al_ivar, cls_func.data_name, 'ivar', workers, batch, s)
        twodpaper(cls_func=cls_func, 
                  Xpl=Xpl, 
                  Ypl=Ypl, 
                  p_test=p_test, 
                  theta0=al_ivar._info['theta0'], 
                  reps0=al_ivar._info['reps0'], 
                  thetainit=theta0,
                  name='branin' + "CL" + '_' + 'ivar' + '.png')
        