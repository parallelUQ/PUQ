import numpy as np
from PUQ.surrogate import emulator
from PUQ.surrogatemethods.PCGPexp import  postpred, postpredbias
from ptest_funcs import pritam
from plots_design import create_test_non, add_result, samplingdata, observe_results
import matplotlib.pyplot as plt
from PUQ.prior import prior_dist


res = []
for s in [0, 1, 2, 3, 4, 5]:
    for nmax in [30, 100, 500, 1000, 5000]:
        
        print('s')
        print(s)
        print('nmax')
        print(nmax)
          
        s = int(s)
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        xr = np.array([[xx, yy] for xx in x for yy in y])
        xr = np.concatenate((xr, xr))
        cls_data = pritam()
        cls_data.realdata(xr, seed=s)
        
        
        
        prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
        prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]) 
        prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[2][0]]), b=np.array([cls_data.thetalimits[2][1]]))
            
        xt_test, ftest, ptest, thetamesh, xmesh = create_test_non(cls_data)
               
        cls_data_y = pritam()
        cls_data_y.realdata(x=xmesh, seed=s)
         
        xt_LHS = samplingdata('LHS', nmax, cls_data, s, prior_xt)
        f_LHS = cls_data.function(xt_LHS[:, 0], xt_LHS[:, 1], xt_LHS[:, 2])
        
        
        x_emu = np.arange(0, 1)[:, None ]
        emu = emulator(x_emu, 
                       xt_LHS, 
                       f_LHS[None, :], 
                       method='PCGPexp')
        
        dx = xmesh.shape[1]
        xtrue_test = [np.concatenate([xc.reshape(1, dx), np.array(cls_data.true_theta).reshape(1, 1)], axis=1) for xc in xmesh]
        xtrue_test = np.array([m for mesh in xtrue_test for m in mesh])
        predobj = emu.predict(x=x_emu, theta=xtrue_test)
        fmeanhat, fvarhat = predobj.mean(), predobj.var()
        
        plt.plot(cls_data_y.real_data.flatten())
        plt.plot(fmeanhat.flatten())    
        plt.show()
        
        print(np.mean(np.abs(fmeanhat.flatten() - cls_data_y.real_data.flatten())))
        
        pmeanhat, pvarhat = postpred(emu._info, cls_data.x, xt_test, cls_data.real_data, cls_data.obsvar)
        
        plt.plot(pmeanhat)
        plt.plot(ptest)
        plt.show()
        
        plt.plot(pmeanhat)
        plt.show()
        post_error = np.mean(np.abs(pmeanhat - ptest))
        print(post_error)
        
        res.append({'posterror': np.mean(np.abs(pmeanhat - ptest)), 
                    'prederror': np.mean(np.abs(fmeanhat.flatten() - cls_data_y.real_data.flatten())),
                    's': s,
                    'nmax': nmax})