import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen, twoD, heatmap
import pandas as pd
from PUQ.design import designer
from smt.sampling_methods import LHS


args = parse_arguments()

# # # # # 
args.funcname = 'banana'
args.seedmin = 1
args.seedmax = 2
# # # # # 

workers = args.minibatch + 1
grid = 10
rep0 = 10
nmesh = 50

dfl = []
if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):
        
        cls_func = eval(args.funcname)()
        cls_func.realdata(seed=s)

        theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
        test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

        heatmap(cls_func)
    
        noise_test = np.zeros(f_test.shape)
        if args.funcname is not 'unimodal':
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
        theta0 = np.repeat(theta0u, rep0, axis=0)
        f0     = np.zeros((cls_func.d, rep0*(grid**2)))
        for i in range(0, rep0*(grid**2)):
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

        
        if args.funcname is not 'unimodal':
            import matplotlib.pyplot as plt
            
            for i in range(0, cls_func.d):
                # plt.hist(fhat[i, :]-f_test[:, i])
                # plt.show()
                
                plt.scatter(fhat[i, :], f_test[:, i])
                plt.xlabel('fhat')
                plt.ylabel('ftest')
                plt.show()
                
                plt.hist(fhat[i, :] - f_test[:, i])
                plt.show()
                
                # plt.scatter(f_test[:, i], fhat[i, :]-f_test[:, i])
                # plt.show()
            
                # plt.plot(emupred._info['nugs'][i, :], color='red')
                # plt.plot(noise_test[:, i], color='blue')
                # plt.show()
                
                plt.scatter(emupred._info['nugs'][i, :], noise_test[:, i])
                plt.xlabel('nugs')
                plt.ylabel('nugtest')
                plt.show()
                
                plt.hist(emupred._info['nugs'][i, :] - noise_test[:, i])
                plt.show()
                
                plt.plot(emupred._info['S'][i, i, :])
                plt.show()
                
                plt.plot(fhat[i, :], color='yellow')
                #plt.plot(fhat[i, :] - np.sqrt(emupred._info['S'][i, i, :]), color='blue')
                #plt.plot(fhat[i, :] + np.sqrt(emupred._info['S'][i, i, :]), color='red')
                plt.show()
                
                plt.plot(fhat[i, :], color='yellow')
                plt.plot(fhat[i, :] - np.sqrt(emupred._info['S'][i, i, :]), color='blue')
                # plt.plot(fhat[i, :] + np.sqrt(emupred._info['S'][i, i, :]), color='red')
                plt.show()
                
                plt.plot(fhat[i, :], color='yellow')
                # plt.plot(fhat[i, :] - np.sqrt(emupred._info['S'][i, i, :]), color='blue')
                plt.plot(fhat[i, :] + np.sqrt(emupred._info['S'][i, i, :]), color='red')
                plt.show()

                
       

        else:
            import matplotlib.pyplot as plt
            plt.scatter(fhat[0, :], f_test)
            plt.xlabel('fhat')
            plt.ylabel('ftest')
            plt.show()
                
            plt.hist(fhat[0, :]-f_test)
            plt.show()
            
            plt.scatter(f_test, fhat[0, :]-f_test)
            plt.show()
            
            plt.plot(emupred._info['nugs'].flatten(), color='red')
            plt.plot(noise_test.flatten(), color='blue')
            plt.show()
            
            plt.scatter(emupred._info['nugs'].flatten(), noise_test.flatten())
            plt.xlabel('nugs')
            plt.ylabel('nugtest')
            plt.show()
            
            # plt.plot(fhat[0, :]])
            plt.plot(emupred._info['S'][0, 0, :])
            plt.show()
            