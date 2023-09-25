import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_post, obsdata, fitemu, create_test, gather_data, add_result, sampling, plot_des, find_mle, samplingnew
from ptest_funcs import sinfunc
import pandas as pd
import seaborn as sns


cls_data = sinfunc()
cls_data.realdata(x=np.array([0, 0.25, 0.5, 0.75, 1])[:, None], seed=0)
# Observe
obsdata(cls_data)
    
args         = parse_arguments()

prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}
    
seeds = 10
ninit = 10
nmax = 60
result = []

for s in range(seeds):


    # # # Create a mesh for test set # # # 
    xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
    nmesh = len(xmesh)
    cls_data_y = sinfunc()
    cls_data_y.realdata(x=xmesh, seed=s)
    ytest = cls_data_y.real_data
    
    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'th': thetamesh,    
                 'xmesh': xmesh,
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
                                 'max_evals': nmax})
    
    xt_eivarx = al_ceivarx._info['theta']
    f_eivarx = al_ceivarx._info['f']
    theta_mle_eivarx = al_ceivarx._info['thetamle'][-1]
    des_eivarx = al_ceivarx._info['des']
    
    plot_des(des_eivarx, xt_eivarx, ninit, cls_data)

    xtrue_test = np.concatenate((xmesh, np.repeat(theta_mle_eivarx, nmesh).reshape(nmesh, theta_mle.shape[1])), axis=1)
    
    phat_eivarx, pvar_eivarx, yhat_eivarx, yvar_eivarx = fitemu(xt_eivarx, f_eivarx[:, None], xt_test, xtrue_test, thetamesh, cls_data) 
    rep = add_result('eivarx', phat_eivarx, ptest, yhat_eivarx, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh, phat_eivarx, ptest, pvar_eivarx)
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
                                 'max_evals': nmax})
    
    xt_eivar = al_ceivar._info['theta']
    f_eivar = al_ceivar._info['f']
    theta_mle_eivar = al_ceivar._info['thetamle'][-1]
    des_eivar = al_ceivar._info['des']
    
    plot_des(des_eivar, xt_eivar, ninit, cls_data)

    xtrue_test = np.concatenate((xmesh, np.repeat(theta_mle_eivar, nmesh).reshape(nmesh, theta_mle.shape[1])), axis=1)
    
    phat_eivar, pvar_eivar, yhat_eivar, yvar_eivar = fitemu(xt_eivar, f_eivar[:, None], xt_test, xtrue_test, thetamesh, cls_data) 
    rep = add_result('eivar', phat_eivar, ptest, yhat_eivar, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh, phat_eivar, ptest, pvar_eivar)
    # # # # # # # # # # # # # # # # # # # # # 
    
    # LHS 
    xt_LHS, f_LHS = samplingnew('LHS', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh)
    thetamle_LHS = find_mle(xt_LHS, f_LHS[None, :], cls_data)
    xtrue_test = np.concatenate((xmesh, np.repeat(thetamle_LHS, nmesh).reshape(nmesh, thetamle_LHS.shape[1])), axis=1)
    phat_lhs, pvar_lhs, yhat_lhs, yvar_lhs = sampling('LHS', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh)
    rep = add_result('lhs', phat_lhs, ptest, yhat_lhs, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh, phat_lhs, ptest, pvar_lhs)
    
    # rnd 
    xt_RND, f_RND = samplingnew('Random', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh)
    thetamle_RND = find_mle(xt_RND, f_RND[None, :], cls_data)
    xtrue_test = np.concatenate((xmesh, np.repeat(thetamle_RND, nmesh).reshape(nmesh, thetamle_RND.shape[1])), axis=1)
    phat_rnd, pvar_rnd, yhat_rnd, yvar_rnd = sampling('Random', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh)
    rep = add_result('rnd', phat_rnd, ptest, yhat_rnd, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh, phat_rnd, ptest, pvar_rnd)

    # Unif
    xt_UNIF, f_UNIF = samplingnew('Uniform', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh)
    thetamle_UNIF = find_mle(xt_UNIF, f_UNIF[None, :], cls_data)
    xtrue_test = np.concatenate((xmesh, np.repeat(thetamle_UNIF, nmesh).reshape(nmesh, thetamle_UNIF.shape[1])), axis=1)
    phat_unif, pvar_unif, yhat_unif, yvar_unif = sampling('Uniform', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh)
    rep = add_result('unif', phat_unif, ptest, yhat_unif, ytest, s)
    result.append(rep)

    plot_post(thetamesh, phat_unif, ptest, pvar_unif)

methods = ['eivar', 'eivarx', 'unif', 'lhs', 'rnd']
methods = ['eivarx',  'lhs', 'rnd']
result_filtered = [r for r in result if r['method'] in methods ]

df = pd.DataFrame(result_filtered)
sns.boxplot(x='method', y='Posterior Error', data=df)
plt.show()
sns.boxplot(x='method', y='Prediction Error', data=df)
plt.show()
