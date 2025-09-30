import numpy as np
from PUQ.prior import prior_dist
from test_funcs import bimodal, banana, unimodal
from smt.sampling_methods import LHS
from PUQ.surrogate import emulator

n0 = 15
rep0 = 2

if __name__ == "__main__":

    for s in np.arange(0, 1):

        cls_func = eval("banana")()
        cls_func.realdata(seed=s)

        # Set a uniform prior
        prior_func = prior_dist(dist="uniform")(
            a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
        )

        # Set random stream for initial design
        persis_info = {"rand_stream": np.random.default_rng(s)}

        # Initial sample
        sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
        theta0 = sampling(n0)
        theta0 = np.repeat(theta0, rep0, axis=0)
        f0 = np.zeros((cls_func.d, n0 * rep0))
        for i in range(0, n0 * rep0):
            f0[:, i] = cls_func.sim_f(theta0[i, :], persis_info=persis_info)
            
        
        d = f0.shape[0]
        md = np.arange(d).reshape(d, 1)
        emu = emulator(x=theta0, 
                       theta=md, 
                       f=f0,                
                       method="multihetGP",
                       args={'lower':None, 'upper':None,
                              'noiseControl':{'k_theta_g_bounds': (1, 100), 'g_max': 1e2, 'g_bounds': (1e-6, 1)}, 
                              'init':{}, 
                              'known':{}, 
                               'settings':{"linkThetas": 'joint', "logN": True, "initStrategy": 'residuals', 
                                         "checkHom": True, "penalty": True, "trace": 0, "return.matrices": True, 
                                         "return.hom": False, "factr": 1e9}})
        
        nmesh = 50
        xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
        ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
        Xpl, Ypl = np.meshgrid(xpl, ypl)
        theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
        pre_obj = emu.predict(x=theta_test)