import numpy as np
import pandas as pd
import scipy.stats as sps
from PUQ.surrogate import emulator
from PUQ.designmethods.allocate_reps import allocate
from PUQ.designmethods.gen_funcs.acquire_new import acquire
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    multiple_pdfs,
    build_emulator,
)



def fit(fitinfo, 
        data_cls,         
        acquisition,
        theta=None,
        prior=None,
        batch_size=None,
        data_test=None,
        nworkers=None,
        max_iter=None,
        des_init={},
        des_add={},
        alloc_settings={},
        pc_settings={'standardize': True, 'latent': True},
        des_settings={},
        **kwargs):


    out = data_cls.out

    gen_specs = {
        "user": {
            "batch_size": batch_size,  # No. of thetas to generate per step
            "nworkers": nworkers,
            "theta_init": des_init.get('theta'),
            "theta_add": des_add.get('theta'),
            "seed_n0": des_init.get('seed'),
            "synth_cls": data_cls,
            "test_data": data_test,
            "prior": prior,
            "n0": des_init.get('n0'),
            "rep0": des_init.get('rep0'),
            "alloc_settings": alloc_settings,
            "acquisition": acquisition,
            "pc_settings": pc_settings,
            "des_settings": des_settings,
        },
    }

    H = gen_f(gen_specs, max_iter)
    
    fitinfo['theta0'] = H['theta0']
    fitinfo['reps0'] = H['reps0']
    fitinfo['theta'] = H['theta']
    fitinfo['f'] = H['f']    
    fitinfo['TV'] = H['TV']
    fitinfo['ivar'] = H['ivar']
    fitinfo['iter_explore'] = H['iter_explore']
    fitinfo['iter_exploit'] = H['iter_exploit']
    return

def newiteration(theta, fevals, x, pc_settings, test_data, obsvar, obs):
    
    thetatest, ptest, ftest, priortest = None, None, None, None
    if test_data is not None:
        thetatest, ptest, ftest, priortest = (
            test_data["theta"],
            test_data["p"],
            test_data["f"],
            test_data["p_prior"],
        )
        
    emu = build_emulator(x, theta, fevals, pc_settings) 
    
    d = len(x)
    obsvar3d = obsvar.reshape(1, d, d) 
    
    # ntest x d
    pred = emu.predict(x=x, theta=thetatest)
    mu = pred.mean().T 
    S = pred._info['S'] 
    St = np.transpose(S, (2, 0, 1))
    N = St + obsvar3d
    phat = multiple_pdfs(obs, mu, N)

    # Obtain the accuracy on the test set
    if ptest is not None:
        TV = np.mean(np.abs(ptest - phat))
    
    return emu, TV
                    
                    
def gen_f(gen_specs, max_iter):
    """Generator to select and obviate parameters for calibration."""

    b = gen_specs["user"]["batch_size"]
    n_workers = gen_specs["user"]["nworkers"]
    seed = gen_specs["user"]["seed_n0"]
    data_cls = gen_specs["user"]["synth_cls"]
    test_data = gen_specs["user"]["test_data"]
    prior_func = gen_specs["user"]["prior"]
    theta_init = gen_specs["user"]["theta_init"]
    theta_add = gen_specs["user"]["theta_add"]
    n0 = gen_specs["user"]["n0"]
    rep0 = gen_specs["user"]["rep0"]
    alloc_settings = gen_specs["user"]["alloc_settings"]
    acqfunc = gen_specs["user"]["acquisition"]
    pc_settings = gen_specs["user"]["pc_settings"]
    des_settings = gen_specs["user"]["des_settings"]

    
    np.random.seed(seed)
    
    # data params
    d = data_cls.d
    x = data_cls.x
    y = data_cls.real_data
    Sigma = data_cls.obsvar
    
    # explore params
    rho = alloc_settings.get('rho')
    if rho is not None:
        b_new = int(b*rho)
        reps_explore = int(b/b_new)
            
    # first iteration
    theta = theta_init[0:n0, :] 
    idx = np.arange(0, n0)[:, None]
    theta = np.repeat(theta, rep0, axis=0)
    idx = np.repeat(idx, rep0, axis=0)
    fevals = np.zeros((d, n0*rep0))
    
    #rng = np.random.default_rng(seed)
    #args = {'rng':rng}
    for i in range(0, n0*rep0):
        fevals[:, i] = data_cls.sim_f(theta[i, :], None) #, **args)
    
    TVlist = []
    counter = 0
    ivarlist = []

    is_explore, is_exploit = des_settings.get('is_explore'), des_settings.get('is_exploit')
    ivar_explore, ivar_exploit = np.inf, np.inf
    iter_explore, iter_exploit = 0, 0
    
    while counter < max_iter:
        emu, TV = newiteration(theta, fevals, x, pc_settings, test_data, Sigma, y)
        TVlist.append(TV)
        
        # predtrue = emu.predict(x=x, theta=data_cls.theta_true).mean()
        # import matplotlib.pyplot as plt
        # plt.plot(predtrue)
        # plt.plot(data_cls.real_data.T)
        # plt.show()
        
        if theta_add is None:
            if is_exploit:
                # Allocate existing ones
                allocate_obj = allocate(budget=b,
                                        emu_info=emu, #emu._info, 
                                        prior=prior_func,
                                        func_cls=data_cls, 
                                        theta_mesh=test_data["theta"], 
                                        method=alloc_settings.get('method'),
                                        alloc_settings=alloc_settings)
                allocate_obj.allocatereps()
                ivar_exploit = allocate_obj.ivar_exploit(emu, pc_settings)
    
                reps_exploit = allocate_obj.reps
                theta_exploit = allocate_obj.theta

                
            if is_explore:
                # Find new ones
                acquire_obj = acquire(bnew=b_new, 
                                      rep=reps_explore,
                                      emu=emu, 
                                      func_cls=data_cls, 
                                      theta_mesh=test_data["theta"], 
                                      prior=prior_func,
                                      method=acqfunc,
                                      nL=500,
                                      pc_settings=pc_settings)
                
                acquire_obj.acquire_new()
                
                ivar_explore = acquire_obj.ivar
                theta_explore = acquire_obj.tnew
    
            ivarlist.append({'exploit':ivar_exploit, 'explore':ivar_explore})
            
            print(ivar_exploit)
            print(ivar_explore)
            if ivar_exploit <= ivar_explore:
                new_theta = np.repeat(theta_exploit, reps_exploit, axis=0)
                iter_exploit += 1
            else:
                new_theta = np.repeat(theta_explore, reps_explore, axis=0)
                iter_explore += 1
                if is_exploit == False:
                    if iter_explore >= (0.5*max_iter):
                        is_exploit, is_explore, ivar_explore = True, False, np.inf
                    
                    
        else:
            cand_theta = theta_add[counter*b_new:(counter + 1)*b_new, :]
            new_theta = np.repeat(cand_theta, reps_explore, axis=0)
                
        new_fevals = np.zeros((d, b))
        for i in range(0, len(new_theta)):
            new_fevals[:, i] = data_cls.sim_f(new_theta[i, :], None) #, **args)
        theta = np.concatenate((theta, new_theta), axis=0)
        fevals = np.concatenate((fevals, new_fevals), axis=1)     
        counter += 1
        
    print('Total exploration:', iter_explore)
    print('Total exploitation:', iter_exploit)
    
    emu, TV = newiteration(theta, fevals, x, pc_settings, test_data, Sigma, y)
    TVlist.append(TV)
    H = {}
    H['theta0'] = emu._info['emulist'][0]['X0']
    H['reps0'] = emu._info['emulist'][0]['mult']
    H['TV'] = TVlist
    H['theta'] = theta
    H['f'] = fevals
    H['ivar'] = ivarlist
    H['iter_explore'] = iter_explore
    H['iter_exploit'] = iter_exploit
    
    return H
