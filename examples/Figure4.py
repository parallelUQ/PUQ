import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen
import pandas as pd
from PUQ.design import designer
from smt.sampling_methods import LHS
import time

args = parse_arguments()

args.seedmin = 0
args.seedmax = 30
args.minibatch = 64
args.funcname = 'banana'

workers = args.minibatch + 1
grid = 10
rep0 = 2
nmesh = 50
rho = 1/2
batch = args.minibatch
maxiter = 192
dftv, dftviter = [], []
if __name__ == "__main__":
    design_start = time.time()
    print("Running function: " + args.funcname)
    for s in np.arange(args.seedmin, args.seedmax):
        print("Running seed no: ", str(s))
        
        cls_func = eval(args.funcname)()
        cls_func.realdata(seed=s)

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
                "trace": 0,
                "des_init":{'seed':s, 'theta':theta0, 'f':f0},
                "alloc_settings":{'method':'ivar', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':False},
                "trace":0,
            },
        )
        
        dftviter.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'ivar', 'example':args.funcname} for t, MAD in enumerate(al_ivar._info['TViter']) if t < maxiter+1])
        dftv.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'ivar', 'example':args.funcname} for t, MAD in enumerate(al_ivar._info['TV'])])
                
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
                "trace": 0,
                "des_init":{'seed':s, 'theta':theta0, 'f':f0},
                "alloc_settings":{'method':'imse', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':False},
                "trace":0,
            },
        )

        dftviter.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'imse', 'example':args.funcname} for t, MAD in enumerate(al_imse._info['TViter']) if t < maxiter+1])
        dftv.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'imse', 'example':args.funcname} for t, MAD in enumerate(al_imse._info['TV'])])
                
        theta0r = theta0u[persis_info['rand_stream'].permutation(theta0u.shape[0]), :]

        al_unif = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition=None,
            args={
                "prior": prior_func,
                "data_test": test_data,
                "max_iter": maxiter,
                "nworkers": workers,
                "batch_size": batch,
                "trace": 0,
                "des_init": {'seed':s, 'theta':theta0, 'f':f0},
                "des_add": {'theta':theta0r},
                "alloc_settings":{'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':True},
                "trace":0,
            },
        )

        dftviter.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'unif', 'example':args.funcname} for t, MAD in enumerate(al_unif._info['TViter']) if t < maxiter+1])
        dftv.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'unif', 'example':args.funcname} for t, MAD in enumerate(al_unif._info['TV'])])
    
    design_end = time.time()
    print("Elapsed time: " + str(round(design_end - design_start, 2)))
    
from summary import lineplot
df = pd.DataFrame(dftv)
lineplot(df, examples=[args.funcname], batches=[batch], save=True)

df = pd.DataFrame(dftviter)
lineplot(df, examples=[args.funcname], batches=[batch], metric='iter', save=True)
