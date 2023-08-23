import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_post, obsdata, fitemu, create_test, gather_data, add_result, sampling, plot_des
from ptest_funcs import bellcurve
import pandas as pd
import seaborn as sns


cls_data = bellcurve()
cls_data.realdata(x=np.array([0, 0.25, 0.5, 0.75, 1])[:, None], seed=0)
# Observe
obsdata(cls_data)
    
args         = parse_arguments()

prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}
    
seeds = 1
ninit = 10
nmax = 60
result = []
for s in range(seeds):


    # # # Create a mesh for test set # # # 
    xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
         
    cls_data_y = bellcurve()
    cls_data_y.realdata(x=xmesh[:, None], seed=0)
    ytest = cls_data_y.real_data
    
    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'th': thetamesh[:, None],    
                 'xmesh': xmesh[:, None],
                 'p_prior': 1} 
    # # # # # # # # # # # # # # # # # # # # # 
    al_ceivarx = designer(data_cls=cls_data, 
                           method='SEQCOMBINEDDES', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivarx',
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': True})
    
    xt_eivarx = al_ceivarx._info['theta']
    f_eivarx = al_ceivarx._info['f']
    theta_mle = al_ceivarx._info['thetamle'][-1]
    des_eivarx = al_ceivarx._info['des']
    
    plot_des(des_eivarx, xt_eivarx, ninit, cls_data)

    xtrue_test = np.concatenate((xmesh[:, None], np.repeat(theta_mle, len(xmesh[:, None])).reshape(len(xmesh[:, None]), theta_mle.shape[1])), axis=1)
    
    phat_eivarx, pvar_eivarx, yhat_eivarx, yvar_eivarx = fitemu(xt_eivarx, f_eivarx[:, None], xt_test, xtrue_test, thetamesh, cls_data) 
    rep = add_result('eivarx', phat_eivarx, ptest, yhat_eivarx, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh[:, None], phat_eivarx, ptest, pvar_eivarx)
    # # # # # # # # # # # # # # # # # # # # # 
    al_ceivar = designer(data_cls=cls_data, 
                           method='SEQCOMBINEDDES', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivar',
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': True})
    
    xt_eivar = al_ceivar._info['theta']
    f_eivar = al_ceivar._info['f']
    theta_mle = al_ceivar._info['thetamle'][-1]
    des_eivar = al_ceivar._info['des']
    

    plot_des(des_eivar, xt_eivar, ninit, cls_data)

    xtrue_test = np.concatenate((xmesh[:, None], np.repeat(theta_mle, len(xmesh[:, None])).reshape(len(xmesh[:, None]), theta_mle.shape[1])), axis=1)
    
    phat_eivar, pvar_eivar, yhat_eivar, yvar_eivar = fitemu(xt_eivar, f_eivar[:, None], xt_test, xtrue_test, thetamesh, cls_data) 
    rep = add_result('eivar', phat_eivar, ptest, yhat_eivar, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh[:, None], phat_eivar, ptest, pvar_eivar)
    # # # # # # # # # # # # # # # # # # # # # 
    
    # LHS 
    phat_lhs, pvar_lhs, yhat_lhs, yvar_lhs = sampling('LHS', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh[:, None])
    rep = add_result('lhs', phat_lhs, ptest, yhat_lhs, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh[:, None], phat_lhs, ptest, pvar_lhs)
    
    # rnd 
    phat_rnd, pvar_rnd, yhat_rnd, yvar_rnd = sampling('Random', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh[:, None])
    rep = add_result('rnd', phat_rnd, ptest, yhat_rnd, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh[:, None], phat_rnd, ptest, pvar_rnd)

    # Unif
    phat_unif, pvar_unif, yhat_unif, yvar_unif = sampling('Uniform', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh[:, None])
    rep = add_result('unif', phat_unif, ptest, yhat_unif, ytest, s)
    result.append(rep)

    plot_post(thetamesh[:, None], phat_unif, ptest, pvar_unif)




df = pd.DataFrame(result)
sns.boxplot(x='method', y='Posterior Error', data=df)
plt.show()
sns.boxplot(x='method', y='Prediction Error', data=df)
plt.show()

