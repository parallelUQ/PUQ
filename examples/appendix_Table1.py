import numpy as np
import matplotlib.pyplot as plt               
import scipy.stats as sps
from PUQ.utils import parse_arguments
from smt.sampling_methods import LHS
from sir_funcs import SIR, SEIRDS
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats.mstats import winsorize


args = parse_arguments()

# # # # # # 
# args.seedmin = 0
# args.seedmax = 10
# # # # # 
examples = ['SIR', 'SEIRDS']

total_reps = 30
r_SS = []

rng = 111
if __name__ == "__main__":
    for example in examples:
        if example == 'SIR':
            n0 = 200
            rep0 = 50
            nmesh = 50
            nt = nmesh**2
            nrep = 1000
            
            # # Create test data
            cls_func = eval('SIR')()
            xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
            ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
            Xpl, Ypl = np.meshgrid(xpl, ypl)
            theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
            
            f_test = np.zeros((nt, cls_func.d))
            f_var = np.zeros((nt, cls_func.d))
            
            persis_info = {'rand_stream': np.random.default_rng(rng)}
            for thid, th in enumerate(theta_test):
                IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
                f_test[thid, :] = np.mean(IrIdRD, axis=0)
                f_var[thid, :]  = np.var(IrIdRD, axis=0)
                plt.plot(np.arange(0, cls_func.d), f_test[thid, :])
            IrIdRDtrue = cls_func.sim_f(thetas=cls_func.theta_true, return_all=True, repl=1000, persis_info=persis_info)
            IrIdRDtrue = np.mean(IrIdRDtrue, axis=0)
            plt.scatter(np.arange(0, cls_func.d), IrIdRDtrue, zorder=2)
            plt.show()
        
        else:
            # # Create test data
            n0 = 200
            rep0 = 50
            nt = 2500
            nrep = 1000
        
            cls_func = eval('SEIRDS')()
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=rng)
            theta_test = sampling(nt)
  
            d = cls_func.d
            f_test = np.zeros((nt, d))
            f_var = np.zeros((nt, d))
        
            persis_info = {'rand_stream': np.random.default_rng(rng)}
            for thid, th in enumerate(theta_test):
                IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
                f_test[thid, :] = np.mean(IrIdRD, axis=0)
                f_var[thid, :]  = np.var(IrIdRD, axis=0)
                plt.plot(np.arange(0, d), f_test[thid, :])
            IrIdRDtrueall = cls_func.sim_f(thetas=cls_func.theta_true, return_all=True, repl=nrep, persis_info=persis_info)
            IrIdRDtrue = np.mean(IrIdRDtrueall, axis=0)
            plt.scatter(np.arange(0, d), IrIdRDtrue, zorder=2)
            plt.show()
        
        def OneRep(example, s):
            dfl = []
            from sir_funcs import SIR, SEIRDS
            from PUQ.surrogate import emulator
            persis_info = {'rand_stream': np.random.default_rng(s)}
            if example == 'SIR':
                cls_func = eval('SIR')()
                cls_func.realdata(seed=s)
                
                p_test = np.zeros(nmesh**2)
                for thid, th in enumerate(theta_test):
                    rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
                    p_test[thid] = rnd.pdf(cls_func.real_data)

            else:
                cls_func = eval('SEIRDS')()
                cls_func.realdata(seed=s)
    
                p_test = np.zeros(nt)
                for thid, th in enumerate(theta_test):
                    rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
                    p_test[thid] = rnd.pdf(cls_func.real_data)
 
            # Initial sample
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
            theta0   = sampling(n0)
            theta0   = np.repeat(theta0, rep0, axis=0)
            f0       = np.zeros((cls_func.d, n0*rep0))
            for i in range(0, n0*rep0):
                f0[:, i] = cls_func.sim_f(theta0[i, :], persis_info=persis_info)
            

            pc_settings = {'standardize': True, 'latent': False}
            
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
                                   'pc_settings':pc_settings})
            
            emupred = emu.predict(x=cls_func.x, theta=theta_test)
            
            fhat = emupred.mean()

            for i in range(0, cls_func.d):
                
                res_p = (f_test[:, i] - fhat[i, :])/(f_test[:, i])
                resnug_p = (f_var[:, i] - emupred._info['nugs'][i, :])/(f_var[:, i])

                mape = np.mean(np.abs(res_p))
                mapenugs = np.mean(np.abs(resnug_p))

                dfl.append({"mape": mape, "mapenugs": mapenugs,
                            "function": example, "j": i+1, "rep": s})
            
            return dfl
                
                
        
        results = Parallel(n_jobs=10)(
            delayed(OneRep)(example, rep_no) for rep_no in range(total_reps)
        )
        r_SS.append(results)
           
    dfl = []
    for rs in r_SS: # example
        for rs_ in rs: # replication
            for rs__ in rs_: # dimension
                dfl.append(rs__)
        

    df = pd.DataFrame(dfl)
    
    print(100*np.round(df[df['function'] == 'SIR'][['mape', 'j']].groupby(['j']).median(), 2))
    print(100*np.round(df[df['function'] == 'SEIRDS'][['mape', 'j']].groupby(['j']).median(), 2))
    
    print(100*np.round(df[df['function'] == 'SIR'][['mapenugs', 'j']].groupby(['j']).median(), 2))
    print(100*np.round(df[df['function'] == 'SEIRDS'][['mapenugs', 'j']].groupby(['j']).median(), 2))