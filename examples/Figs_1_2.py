import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, obsdata, create_test, add_result, samplingdata
from ptest_funcs import sinfunc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

args = parse_arguments()

s = 1
ninit = 10
nmax = 30
result = []


cls_data = sinfunc()
dt = len(cls_data.true_theta)
cls_data.realdata(x=np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None], seed=s)
# Observe
#obsdata(cls_data)
        
prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

# # # Create a mesh for test set # # # 
xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
nmesh = len(xmesh)
cls_data_y = sinfunc()
cls_data_y.realdata(x=xmesh, seed=s)
ytest = cls_data_y.real_data

test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'y': ytest,
             'th': thetamesh,    
             'xmesh': xmesh,
             'p_prior': 1} 
# # # # # # # # # # # # # # # # # # # # # 
al_ceivarx = designer(data_cls=cls_data, 
                       method='SEQCOMPDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'ceivarx',
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax})

xt_eivarx = al_ceivarx._info['theta']
f_eivarx = al_ceivarx._info['f']
thetamle_eivarx = al_ceivarx._info['thetamle'][-1]

al_ceivar = designer(data_cls=cls_data, 
                       method='SEQCOMPDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'ceivar',
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax})

xt_eivar = al_ceivar._info['theta']
f_eivar = al_ceivar._info['f']
thetamle_eivar = al_ceivar._info['thetamle'][-1]

# LHS 
xt_LHS, f_LHS = samplingdata('LHS', nmax, cls_data, s, prior_xt)
al_LHS = designer(data_cls=cls_data, 
                       method='SEQGIVEN', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax,
                             'theta_torun': xt_LHS,
                             'bias': False})
xt_LHS = al_LHS._info['theta']
f_LHS = al_LHS._info['f']
thetamle_LHS = al_LHS._info['thetamle'][-1]

dataset = []
dataset.append({'mle':thetamle_eivarx, 'f':f_eivarx, 'xt':xt_eivarx})
dataset.append({'mle':thetamle_eivar, 'f':f_eivar, 'xt':xt_eivar})
dataset.append({'mle':thetamle_LHS, 'f':f_LHS, 'xt':xt_LHS})

xt_true = [np.concatenate([xc.reshape(1, 1), cls_data.true_theta.reshape(1, 1)], axis=1) for xc in xmesh]
xt_true = np.array([m for mesh in xt_true for m in mesh])
true_model = cls_data.function(xt_true[:, 0], xt_true[:, 1])
    
from PUQ.surrogate import emulator
from PUQ.surrogatemethods.PCGPexp import  postpred
x_emu = np.arange(0, 1)[:, None ]

for point in dataset:
    theta_mle = point['mle']
    f = point['f']
    xt = point['xt']
    
    xt_ref = [np.concatenate([xc.reshape(1, 1), theta_mle], axis=1) for xc in xmesh]
    xt_ref = np.array([m for mesh in xt_ref for m in mesh])
     
    emu = emulator(x_emu, 
                   xt, 
                   f[None, :], 
                   method='PCGPexp')
    
    # Predictions
    emupred = emu.predict(x=x_emu, theta=xt_ref)
    predmean = emupred.mean().flatten()
    predsd = np.sqrt(emupred.var().flatten())
    
    plt.plot(xmesh.flatten(), predmean, color='blue', linestyle='dashed', linewidth=2.5) 
    plt.fill_between(xmesh.flatten(), predmean - predsd, predmean + predsd, color='blue', alpha=0.1)
    plt.plot(xmesh.flatten(), true_model.flatten(), color='red', linewidth=2.5) 
    plt.scatter(cls_data.x.flatten(), cls_data.real_data, color='black')
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$\eta(x, \hat{\theta})$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
    # Posterior
    pmeanhat, pvarhat = postpred(emu._info, cls_data.x, xt_test, cls_data.real_data, cls_data.obsvar)
    plt.plot(thetamesh.flatten(), pmeanhat, color='blue', linestyle='dashed', linewidth=2.5) 
    plt.fill_between(thetamesh.flatten(), 
                     pmeanhat - np.sqrt(pvarhat),  
                     pmeanhat + np.sqrt(pvarhat), 
                     color='blue', alpha=0.2)
    plt.plot(thetamesh.flatten(), ptest.flatten(), color='red', linewidth=2.5) 
    plt.ylabel(r'$p(y|\theta)$', fontsize=20)
    plt.xlabel(r'$\theta$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(np.arange(0, 1.5, 0.3), fontsize=15)
    plt.show()

    # Design

    plt.scatter(xt[0:ninit, 0], xt[0:ninit, 1], marker='*', color='blue', s=50)
    plt.scatter(xt[:, 0][ninit:], xt[:, 1][ninit:], marker='+', color='red', s=50)
    plt.axhline(y = cls_data.true_theta, color = 'green')
    plt.scatter(cls_data.x, np.repeat(cls_data.true_theta, len(cls_data.x)), marker='x', color='black', s=50)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$\theta$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
