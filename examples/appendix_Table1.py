import numpy as np
from PUQ.prior import prior_dist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.stats as sps
from PUQ.design import designer
from PUQ.utils import parse_arguments, save_output
from smt.sampling_methods import LHS
from utilities import twoD
from sir_funcs import SIR, SEIRDS
import pandas as pd
import seaborn as sns

args = parse_arguments()

# # # # # 
args.seedmin = 0
args.seedmax = 10
# # # # # 
examples = ['SIR', 'SEIRDS']
dfl = []

if __name__ == "__main__":
    for example in examples:
        if example == 'SIR':
            n0 = 100
            rep0 = 50
            nmesh = 50
            nt = nmesh**2
            
            # # Create test data
            nrep = 1000
            cls_func = eval('SIR')()
            xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
            ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
            Xpl, Ypl = np.meshgrid(xpl, ypl)
            theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
            
            f_test = np.zeros((nt, cls_func.d))
            f_var = np.zeros((nt, cls_func.d))
            
            persis_info = {'rand_stream': np.random.default_rng(100)}
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
            n0 = 100
            rep0 = 100
            nt = 2500
            nrep = 3000
        
            cls_func = eval('SEIRDS')()
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=100)
            theta_test = sampling(nt)
            theta_test[-1,:] = cls_func.theta_true
            d = cls_func.d
            f_test = np.zeros((nt, d))
            f_var = np.zeros((nt, d))
        
            persis_info = {'rand_stream': np.random.default_rng(100)}
            for thid, th in enumerate(theta_test):
                IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
                f_test[thid, :] = np.mean(IrIdRD, axis=0)
                f_var[thid, :]  = np.var(IrIdRD, axis=0)
                plt.plot(np.arange(0, d), f_test[thid, :])
            IrIdRDtrueall = cls_func.sim_f(thetas=cls_func.theta_true, return_all=True, repl=nrep, persis_info=persis_info)
            IrIdRDtrue = np.mean(IrIdRDtrueall, axis=0)
            plt.scatter(np.arange(0, d), IrIdRDtrue, zorder=2)
            plt.show()
        
    
    
        for s in np.arange(args.seedmin, args.seedmax):
            
            if example == 'SIR':
                cls_func = eval('SIR')()
                cls_func.realdata(seed=s)
                
                p_test = np.zeros(nmesh**2)
                for thid, th in enumerate(theta_test):
                    rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
                    p_test[thid] = rnd.pdf(cls_func.real_data)
                
                test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}
                
            else:
                cls_func = eval('SEIRDS')()
                cls_func.realdata(seed=s)
    
                p_test = np.zeros(nt)
                for thid, th in enumerate(theta_test):
                    rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
                    p_test[thid] = rnd.pdf(cls_func.real_data)
                    
            # Set a uniform prior
            prior_func = prior_dist(dist="uniform")(
                a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
            )
            
            plt.plot(np.arange(0, len(IrIdRDtrue)), IrIdRDtrue)
            plt.scatter(np.arange(0, len(IrIdRDtrue)), cls_func.real_data)  
            plt.show()
            
            # Initial sample
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
            theta0   = sampling(n0)
            theta0   = np.repeat(theta0, rep0, axis=0)
            f0       = np.zeros((cls_func.d, n0*rep0))
            for i in range(0, n0*rep0):
                f0[:, i] = cls_func.sim_f(theta0[i, :], persis_info=persis_info)
            
            from PUQ.surrogate import emulator
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
                
                r2 = 1 - sum((fhat[i, :] - f_test[:, i])**2)/sum((f_test[:, i] - np.mean(f_test[:, i]))**2)
                r2nugs = 1 - sum((emupred._info['nugs'][i, :] - f_var[:, i])**2)/sum((f_var[:, i] - np.mean(f_var[:, i]))**2)
                dfl.append({"r2": r2, "r2nugs": r2nugs, "function": example, "j": i+1, "rep": s})
                        
                    
    ft = 20
    df = pd.DataFrame(dfl)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create a continuous cubehelix colormap
    cmap = sns.cubehelix_palette(as_cmap=True)
    
    # Extract unique hue levels
    unique_hues = df['j'].unique()
    n_hues = len(unique_hues)
    
    # Sample colors around the middle of the colormap (values between 0.4 and 0.6)
    middle_range = np.linspace(0.1, 0.9, n_hues)
    discrete_colors = [cmap(value) for value in middle_range]
    
    # Create a mapping of hue levels to sampled colors
    palette = dict(zip(unique_hues, discrete_colors))
    
    sns.boxplot(x='function', y='r2', hue='j', data=df, ax=ax, showfliers=False, palette=palette)
    ax.set_xlabel("Example", fontsize=ft)
    ax.set_ylabel(r'$r^2$ (Emulator mean)', fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    lgd = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.4),
              fancybox = True, shadow = True, ncol = 3, fontsize=ft-5, title="j")
    lgd.get_title().set_fontsize(ft-5)
    #ax.set_ylim(0.75, 1.1)
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.boxplot(x='function', y='r2nugs', hue='j', data=df, ax=ax, showfliers=False, palette=palette)
    ax.set_xlabel("Example", fontsize=ft)
    ax.set_ylabel(r'$r^2$ (Intrinsic noise)', fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    lgd = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.4),
              fancybox = True, shadow = True, ncol = 3, fontsize=ft-5, title="j")
    #ax.set_ylim(0.75, 1.1)
    lgd.get_title().set_fontsize(ft-5)
    plt.show()
    
    print(np.round(df[df['function'] == 'SIR'][['r2', 'j']].groupby(['j']).median(), 2))
    print(np.round(df[df['function'] == 'SEIRDS'][['r2', 'j']].groupby(['j']).median(), 2))
    
    print(np.round(df[df['function'] == 'SIR'][['r2nugs', 'j']].groupby(['j']).median(), 2))
    print(np.round(df[df['function'] == 'SEIRDS'][['r2nugs', 'j']].groupby(['j']).median(), 2))