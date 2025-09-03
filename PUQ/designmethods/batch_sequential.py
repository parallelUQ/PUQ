import numpy as np
from PUQ.designmethods.gen_funcs.batch_allocate_reps import allocate
from PUQ.designmethods.gen_funcs.acquire_new import acquire
from PUQ.designmethods.gen_funcs.batch_acquisition_funcs_support import (
    multiple_pdfs,
    build_emulator,
)
from joblib import Parallel, delayed
import copy

class batch_sequential_design:
    def __init__(self, cls_func, trace=True):
        self.cls_func = cls_func
        self.trace = trace
        self.y = self.cls_func.real_data
        self.x = self.cls_func.x
        self.d = self.cls_func.d
        self.Sigma = self.cls_func.obsvar
        self.Sigma3d = self.Sigma.reshape(1, self.d, self.d)
        self.detSigma = np.linalg.det(self.Sigma)
        return

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def get(self, key):
        return self.__dict__.get(key)


                     
    def build_design(self, t0, f0, af, args={}):
        # data params
        # d = data_cls.d
        # x = data_cls.x
        # y = data_cls.real_data
        # Sigma = data_cls.obsvar

        
        b = args["batch_size"]
        max_iter = args["max_iter"]
        alloc_settings = args["alloc_settings"]
        des_settings = args["des_settings"]
        pc_settings = args["pc_settings"]
        test_data = args["data_test"]
        prior_func = args["prior"]
        # explore params
        rho = alloc_settings.get('rho')
        if rho is not None:
            b_new = int(b*rho)
            r_new = int(b/b_new)
            
        seed = 1

        persis_info = {"rand_stream": np.random.default_rng(1)}
        
        persis_info = {i: {} for i in range(b)} # add_unique_random_streams({}, b)
        for perid, per in enumerate(persis_info):
            persis_info[perid]['rand_stream'] = np.random.default_rng(seed*b + perid)
            
        rand_stream = np.random.default_rng(seed*(b+1)) #np.random.Generator(np.random.PCG64())

        f_c, t_c = f0, t0
        
        print("here")
        print(f_c.shape)
        print(t_c.shape)
        
        iter_explore, iter_exploit = 0, 0
        is_explore, is_exploit = des_settings.get('is_explore'), des_settings.get('is_exploit')

        for iteration in range(max_iter):
        
            emu, TV = self.newiteration(t_c, f_c, self.x, pc_settings, test_data, self.Sigma, self.y)
            
            theta, iter_exploit, iter_explore = self.onebatch(b, 
                                                         b_new, 
                                                         r_new, 
                                                         emu, 
                                                         prior_func, 
                                                         af, 
                                                         self.cls_func, 
                                                         test_data, 
                                                         alloc_settings, 
                                                         pc_settings, 
                                                         des_settings,
                                                         iter_exploit, 
                                                         iter_explore,
                                                         is_exploit,
                                                         is_explore,
                                                         rand_stream,
                                                         self.trace)
            
            
            fevals = Parallel(n_jobs=b)(
                delayed(self.cls_func.sim_f)(theta[i], persis_info[i]) for i in range(b)
            )
            print(fevals)
            fevals = np.array(fevals).T
            print("here 2")

            f_c = np.concatenate((f_c, fevals), axis=1)
            t_c = np.concatenate((t_c, theta), axis=0)
            
            print(f_c[:, -4:])
            print(t_c.shape)
            


        self.f = f_c
        self.theta = t_c


    def newiteration(self, theta, fevals, x, pc_settings, test_data, obsvar, obs):

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

    def onebatch(self, 
                 b, 
                 b_new, 
                 r_new, 
                 emu, 
                 prior_func, 
                 acqfunc, 
                 data_cls, 
                 test_data, 
                 alloc_settings, 
                 pc_settings, 
                 des_settings,
                 iter_exploit, 
                 iter_explore,
                 is_exploit,
                 is_explore,
                 rand_stream,
                 trace):
    
        ivar_exploit, ivar_explore = np.inf, np.inf
        
        emu_original_info = copy.deepcopy(emu._info)
        
        if is_exploit:
            # Allocate existing ones
            allocate_obj = allocate(budget=b,
                                    emu_info=emu,
                                    prior=prior_func,
                                    func_cls=data_cls, 
                                    theta_mesh=test_data["theta"], 
                                    method=alloc_settings.get('method'),
                                    alloc_settings=alloc_settings,
                                    rand_stream=rand_stream,
                                    trace=trace)
            allocate_obj.allocatereps()
            if not is_explore:
                ivar_exploit = 0
            else:
                ivar_exploit = allocate_obj.ivar_exploit(emu, pc_settings)
        
            r_exploit = allocate_obj.reps
            theta_exploit = allocate_obj.theta
            
        emu._info = emu_original_info       
    
        if is_explore:
            # Find new ones
            acquire_obj = acquire(bnew=b_new, 
                                  rep=r_new,
                                  emu=emu, 
                                  func_cls=data_cls, 
                                  theta_mesh=test_data["theta"], 
                                  prior=prior_func,
                                  method=acqfunc,
                                  nL=des_settings.get('nL'),
                                  pc_settings=pc_settings,
                                  rand_stream=rand_stream,
                                  impute_str=des_settings.get('impute_str'),
                                  skip=not is_exploit)
            
            acquire_obj.acquire_new()
        
            ivar_explore = acquire_obj.ivar
            theta_explore = acquire_obj.tnew
    
        if ivar_exploit <= ivar_explore:
            new_theta = np.repeat(theta_exploit, r_exploit, axis=0)
            iter_exploit += 1
        else:
            new_theta = np.repeat(theta_explore, r_new, axis=0)
            iter_explore += 1
    
        return new_theta, iter_exploit, iter_explore
    