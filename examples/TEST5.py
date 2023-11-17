
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import numpy as np
from PUQ.surrogate import emulator
from plots_design import create_test_non, add_result, samplingdata, observe_results
from ptest_funcs import pritam
from PUQ.surrogatemethods.PCGPexp import  postpred, postpredbias

def plotresult(path, out, ex_name, w, b, rep, method, n0, nf):


    design_saved = read_output(path + out + '/', ex_name, method, w, b, rep)

    theta = design_saved._info['theta']
    f = design_saved._info['f']
    theta_mle = design_saved._info['thetamle']

    return theta, f, theta_mle

# choose either 'pritam' or 'sinfunc'
ex = 'pritam'
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
path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/newPUQ/examples/'
#path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/denoise/' 
result = []
for m in method:  
    print(m)
    for r in np.arange(0, rep):
        
        theta, f, thetamle = plotresult(path, outs, ex, worker, batch, r, m, n0=n0, nf=nf)
        print(thetamle[0])
        x_emu = np.arange(0, 1)[:, None ]
        
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        xr = np.array([[xx, yy] for xx in x for yy in y])
        xr = np.concatenate((xr, xr))
        cls_data = pritam()
        cls_data.realdata(xr, seed=r, isbias=True)
    
        xt_test, ftest, ptest, thetamesh, xmesh = create_test_non(cls_data, is_bias=True)
        #ytest = cls_data.function(xmesh[:, 0], xmesh[:, 1], cls_data.true_theta).reshape(1, len(xmesh))
        ytest = cls_data.function(xmesh[:, 0], xmesh[:, 1], cls_data.true_theta).reshape(1, len(xmesh)) + cls_data.bias(xmesh[:, 0], xmesh[:, 1]).reshape(1, len(xmesh))
       
        
        res_pred_array = np.zeros(len(thetamle))
        res_post_array = np.zeros(len(thetamle))
        
        for tmle_id, tmle in enumerate(thetamle):
            
            thetatr = theta[0:(n0 + tmle_id), :]
            ft = f[0:(n0 + tmle_id)]
            dx = xmesh.shape[1]
            xtrue_test = [np.concatenate([xc.reshape(1, 2), tmle], axis=1) for xc in xmesh]
            xtrue_test = np.array([m for mesh in xtrue_test for m in mesh])
            
            emu = emulator(x_emu, 
                           thetatr, 
                           ft[:, None], 
                           method='PCGPexp')
            
            predobj = emu.predict(x=x_emu, theta=xtrue_test)
            fmeanhat, fvarhat = predobj.mean(), predobj.var()
            
            from sklearn.linear_model import LinearRegression
            xp = [np.concatenate([xc.reshape(1, 2), tmle], axis=1) for xc in xr]
            xp = np.array([m for mesh in xp for m in mesh])
            emupred = emu.predict(x=x_emu, theta=xp)
            emumean = emupred.mean()
            bias = (cls_data.real_data - emumean).T
            model = LinearRegression().fit(xr, bias)
            
            bmeanhat = model.predict(xmesh)
            pred_error = np.mean(np.abs(fmeanhat + bmeanhat.T - ytest))
            
            #plt.plot((fmeanhat + bmeanhat.T).flatten())
            #plt.plot(ytest.flatten())
            #plt.show()
            res_pred_array[tmle_id] = pred_error

            res_post_array[tmle_id] = 0
            
            
            
            #plt.plot(ytest.flatten())
            #plt.plot(fmeanhat.flatten())
            #plt.show()
            
            #plt.plot(ptest.flatten())
            #plt.plot(pmeanhat.flatten())
            #plt.show()
            
        res = {'method': m, 'repno': r, 'Prediction Error': res_pred_array, 'Posterior Error': res_post_array}
        result.append(res)
    
        
#method = ['eivarx', 'eivar', 'lhs', 'rnd']
observe_results(result, method, rep, n0-30, nf-30)
    