
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import numpy as np
from PUQ.surrogate import emulator
from plots_design import create_test, add_result, samplingdata, observe_results
from ptest_funcs import sinfunc
from PUQ.surrogatemethods.PCGPexp import  postpred, postpredbias

def plotresult(path, out, ex_name, w, b, rep, method, n0, nf):


    design_saved = read_output(path + out + '/', ex_name, method, w, b, rep)

    theta = design_saved._info['theta']
    f = design_saved._info['f']
    theta_mle = design_saved._info['thetamle']

    return theta, f, theta_mle

# choose either 'pritam' or 'sinfunc'
ex = 'sinfunc'
is_bias = False
if ex == 'pritam':
    n0, nf = 30, 180
    if is_bias:
        outs = 'pritam_bias'
        method = ['ceivarxbias', 'ceivarbias', 'lhs', 'rnd']
    else:
        outs = 'pritam'     
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
        
elif ex == 'sinfunc':
    n0, nf = 10, 100
    if is_bias:
        outs = 'sinf_bias'
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
    else:
        outs = 'sinf'    
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']

batch = 1
worker = 2
fonts = 18
rep = 1 
path = '/Users/ozgesurer/Desktop/des_examples/newPUQ/examples/'
result = []
for m in method:  
    print(m)
    for r in np.arange(0, rep):
        
        theta, f, thetamle = plotresult(path, outs, ex, worker, batch, r, m, n0=n0, nf=nf)
        print(thetamle[0])
        x_emu = np.arange(0, 1)[:, None ]
        
        cls_data = sinfunc()
        dt = len(cls_data.true_theta)
        cls_data.realdata(x=np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None], seed=r)
        

        # # # Create a mesh for test set # # # 
        xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
        nmesh = len(xmesh)
        cls_data_y = sinfunc()
        cls_data_y.realdata(x=xmesh, seed=r)
        ytest = cls_data.function(xmesh, cls_data.true_theta).reshape(1, 100)#cls_data_y.real_data
        
        res_pred_array = np.zeros(len(thetamle))
        res_post_array = np.zeros(len(thetamle))
        
        for tmle_id, tmle in enumerate(thetamle):
            
            thetatr = theta[0:(n0 + tmle_id), :]
            ft = f[0:(n0 + tmle_id)]
            dx = xmesh.shape[1]
            xtrue_test = [np.concatenate([xc.reshape(1, 1), tmle], axis=1) for xc in xmesh]
            xtrue_test = np.array([m for mesh in xtrue_test for m in mesh])
            
            emu = emulator(x_emu, 
                           thetatr, 
                           ft[:, None], 
                           method='PCGPexp')
            
            predobj = emu.predict(x=x_emu, theta=xtrue_test)
            fmeanhat, fvarhat = predobj.mean(), predobj.var()
            
            pred_error = np.mean(np.abs(fmeanhat - ytest))
            res_pred_array[tmle_id] = pred_error
            
            pmeanhat, pvarhat = postpred(emu._info, cls_data.x, xt_test, cls_data.real_data, cls_data.obsvar)
            post_error = np.mean(np.abs(pmeanhat - ptest))
            res_post_array[tmle_id] = post_error
            
            plt.plot(ytest.flatten())
            plt.plot(fmeanhat.flatten())
            plt.show()
            
            plt.plot(ptest.flatten())
            plt.plot(pmeanhat.flatten())
            plt.show()
            
        res = {'method': m, 'repno': r, 'Prediction Error': res_pred_array, 'Posterior Error': res_post_array}
        result.append(res)
    
        
#method = ['eivarx', 'eivar', 'lhs', 'rnd']
observe_results(result, method, rep, n0-10, nf-10)
    