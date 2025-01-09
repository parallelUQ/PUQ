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
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS

args = parse_arguments()

# # # # # 
funcname = ['unimodal', 'banana', 'bimodal']
args.seedmin = 0
args.seedmax = 10
# # # # # 

workers = args.minibatch + 1
grid = 10
rep0 = 15
nmesh = 50
n0 = grid**2

dfl = []
if __name__ == "__main__":
    for f in funcname:
        for s in np.arange(args.seedmin, args.seedmax):
            
            cls_func = eval(f)()
            cls_func.realdata(seed=0)
    
            theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
            test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}
    
            heatmap(cls_func)
        
            noise_test = np.zeros(f_test.shape)
            if f is not 'unimodal':
                for tid, t in enumerate(theta_test):
                    noise_test[tid, :] = cls_func.noise(t[None, :]).flatten()
            else:
                for tid, t in enumerate(theta_test):
                    noise_test[tid] = cls_func.noise(t[None, :]).flatten()
                
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

            # sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
            # theta0u = sampling(n0)
            
            theta0 = np.repeat(theta0u, rep0, axis=0)
            f0     = np.zeros((cls_func.d, rep0*n0))
            for i in range(0, rep0*n0):
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
    
            
            if f is not 'unimodal':
                for i in range(0, cls_func.d):
                    r2 = 1 - sum((fhat[i, :] - f_test[:, i])**2)/sum((f_test[:, i] - np.mean(f_test[:, i]))**2)
                    r2nugs = 1 - sum((emupred._info['nugs'][i, :] - noise_test[:, i])**2)/sum((noise_test[:, i] - np.mean(noise_test[:, i]))**2)
                    dfl.append({"r2": r2, "r2nugs": r2nugs, "function": f, "j": i+1, "rep": s})
                    
                    print(r2nugs)

            else:
                r2 = 1 - sum((fhat[0, :] - f_test)**2)/sum((f_test - np.mean(f_test))**2)
                r2nugs = 1 - sum((emupred._info['nugs'].flatten() - noise_test.flatten())**2)/sum((noise_test.flatten() - np.mean(noise_test.flatten()))**2)
                dfl.append({"r2": r2, "r2nugs": r2nugs, "function": f, "j": 1, "rep": s})                    
    
            
    import pandas as pd
    import seaborn as sns



    ft = 20
    df = pd.DataFrame(dfl)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create a continuous cubehelix colormap
    cmap = sns.cubehelix_palette(as_cmap=True)
    
    # Extract unique hue levels
    unique_hues = df['j'].unique()
    n_hues = len(unique_hues)
    
    # Sample colors around the middle of the colormap (values between 0.4 and 0.6)
    middle_range = np.linspace(0.2, 0.6, n_hues)
    discrete_colors = [cmap(value) for value in middle_range]
    
    # Create a mapping of hue levels to sampled colors
    palette = dict(zip(unique_hues, discrete_colors))
    
    sns.boxplot(x='function', y='r2', hue='j', data=df, ax=ax, showfliers=False, palette=palette)
    ax.set_xlabel("Example", fontsize=ft)
    ax.set_ylabel(r'$r^2$ (Emulator mean)', fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    lgd = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3),
              fancybox = True, shadow = True, ncol = 2, fontsize=ft-5)
    #ax.set_ylim(0.75, 1.1)
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.boxplot(x='function', y='r2nugs', hue='j', data=df, ax=ax, showfliers=False, palette=palette)
    ax.set_xlabel("Example", fontsize=ft)
    ax.set_ylabel(r'$r^2$ (Intrinsic noise)', fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    lgd = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3),
              fancybox = True, shadow = True, ncol = 2, fontsize=ft-5)
    #ax.set_ylim(0.75, 1.1)
    plt.show()
        