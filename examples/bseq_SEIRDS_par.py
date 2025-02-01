import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from utilities import test_data_gen
from smt.sampling_methods import LHS
from joblib import Parallel, delayed
import os
import time
import scipy.stats as sps
import matplotlib.pyplot as plt
from sir_funcs import SEIRDS
            
args = parse_arguments()


n0 = 50
rep0 = 2
rho = 1/2
maxiter = 384


# # Create test data
nt = 2500
nrep = 1000

cls_func = eval('SEIRDS')()
sampling = LHS(xlimits=cls_func.thetalimits, random_state=100)
theta_test = sampling(nt)
d = cls_func.d
f_test = np.zeros((nt, d))
f_var = np.zeros((nt, d))

persis_info = {}
persis_info['rand_stream'] = np.random.default_rng(100)

for thid, th in enumerate(theta_test):
    IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
    f_test[thid, :] = np.mean(IrIdRD, axis=0)
    f_var[thid, :]  = np.var(IrIdRD, axis=0)
    plt.plot(np.arange(0, d), f_test[thid, :])
IrIdRDtrueall = cls_func.sim_f(thetas=cls_func.theta_true, return_all=True, repl=nrep, persis_info=persis_info)
IrIdRDtrue = np.mean(IrIdRDtrueall, axis=0)
#print(np.round(np.var(IrIdRDtrueall, axis=0), 3))
plt.scatter(np.arange(0, d), IrIdRDtrue, zorder=2)
plt.show()

total_reps = 30

if __name__ == "__main__":
    design_start = time.time()
    print("Running function: " + 'SEIRDS')

    cd = os.getcwd()  
    id_ex = 5
    dp = str(id_ex) + "_" + 'SEIRDS' + "_" + "explore"
    if not os.path.exists(dp):
        os.makedirs(dp)
    fp = os.path.join(cd, dp)
    
    for batch in [8, 16, 32, 64]:
        workers = batch + 1
        
        sd = os.path.join(fp, str(batch))
        if not os.path.isdir(sd):
            os.makedirs(sd)
        sd = os.path.join(sd, "output/")
        if not os.path.isdir(sd):
            os.makedirs(sd)
        
        def OneRep(s, batch, workers):  
            from sir_funcs import SEIRDS
            cls_func = eval('SEIRDS')()
            cls_func.realdata(seed=s)
            
            p_test = np.zeros(nt)
            for thid, th in enumerate(theta_test):
                rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
                p_test[thid] = rnd.pdf(cls_func.real_data)
            
            test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}
            
            # Set a uniform prior
            prior_func = prior_dist(dist="uniform")(
                a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
            )
                      
            # Initial sample
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
            theta0   = sampling(n0)
            theta0   = np.repeat(theta0, rep0, axis=0)
            f0       = np.zeros((cls_func.d, n0*rep0))
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
                    "des_settings":{'is_exploit':True, 'is_explore':True, 'nL': 500, 'impute_str': 'update'}
                },
            )

            save_output(al_ivar, cls_func.data_name, 'ivar', workers, batch, s, sd)
    
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
                    "des_settings":{'is_exploit':True, 'is_explore':True, 'nL': 500, 'impute_str': 'update'}
                },
            )
            
            save_output(al_imse, cls_func.data_name, 'imse', workers, batch, s, sd)

            
            rholhs = 1/4
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
            theta_lhs = sampling(int(maxiter*rholhs))
    
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
                    "des_init":{'seed':s, 'theta':theta0, 'f':f0},
                    "des_add":{'theta':theta_lhs},
                    "alloc_settings":{'use_Ki':True, 'rho': rholhs, 'theta':None, 'a0':None, 'gen':False},
                    "pc_settings":{'standardize':True, 'latent':False},
                    "des_settings":{'is_exploit':True, 'is_explore':True}
                },
            )
            
            save_output(al_unif, cls_func.data_name, 'unif', workers, batch, s, sd)

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
                    "des_settings":{'is_exploit':True, 'is_explore':True, 'nL': 500, 'impute_str': 'update'}
                },
            )

            save_output(al_var, cls_func.data_name, 'var', workers, batch, s, sd)
            
        Parallel(n_jobs=10)(delayed(OneRep)(rep_no, batch, workers) for rep_no in range(0, total_reps))   
    
    design_end = time.time()
    print("Elapsed time: " + str(round(design_end - design_start, 2)))
