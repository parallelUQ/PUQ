import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_post, obsdata, fitemu, create_test, gather_data, add_result, sampling, plot_des, find_mle, samplingdata
from ptest_funcs import multicurve
import pandas as pd
import seaborn as sns



seeds = 1
ninit = 10
nmax = 50
result = []

for s in range(seeds):

    cls_data = multicurve()
    dt = len(cls_data.true_theta)
    cls_data.realdata(x=np.array([0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1])[:, None], seed=s)
    # Observe
    obsdata(cls_data)
        
    args         = parse_arguments()
    
    prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
    prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
    prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))
    
    priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}
    
    # # # Create a mesh for test set # # # 
    xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
    nmesh = len(xmesh)
    cls_data_y = multicurve()
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
    
    plot_EIVAR(xt_eivarx, cls_data, ninit, xlim1=0, xlim2=1)

    res = {'method': 'eivarx', 'repno': s, 'Prediction Error': al_ceivarx._info['TV'], 'Posterior Error': al_ceivarx._info['HD']}
    result.append(res)
    # # # # # # # # # # # # # # # # # # # # # 
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

    plot_EIVAR(xt_eivar, cls_data, ninit, xlim1=0, xlim2=1)

    res = {'method': 'eivar', 'repno': s, 'Prediction Error': al_ceivar._info['TV'], 'Posterior Error': al_ceivar._info['HD']}
    result.append(res)
    
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
                                 'theta_torun': xt_LHS})
    xt_LHS = al_LHS._info['theta']
    f_LHS = al_LHS._info['f']
    thetamle_LHS = al_LHS._info['thetamle'][-1]

    res = {'method': 'lhs', 'repno': s, 'Prediction Error': al_LHS._info['TV'], 'Posterior Error': al_LHS._info['HD']}
    result.append(res)
    
    # rnd 
    xt_RND, f_RND = samplingdata('Random', nmax, cls_data, s, prior_xt)
    al_RND = designer(data_cls=cls_data, 
                           method='SEQGIVEN', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'theta_torun': xt_RND})
    xt_RND = al_RND._info['theta']
    f_RND = al_RND._info['f']
    thetamle_RND = al_RND._info['thetamle'][-1]
    
    res = {'method': 'rnd', 'repno': s, 'Prediction Error': al_RND._info['TV'], 'Posterior Error': al_RND._info['HD']}
    result.append(res)
    
    # Unif
    xt_UNIF, f_UNIF = samplingdata('Uniform', nmax, cls_data, s, prior_xt)
    al_UNIF = designer(data_cls=cls_data, 
                           method='SEQGIVEN', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'theta_torun': xt_UNIF})
    xt_UNIF = al_UNIF._info['theta']
    f_UNIF = al_UNIF._info['f']
    thetamle_UNIF = al_UNIF._info['thetamle'][-1]
    
    res = {'method': 'unif', 'repno': s, 'Prediction Error': al_UNIF._info['TV'], 'Posterior Error': al_UNIF._info['HD']}
    result.append(res)

cols = ['blue', 'red', 'cyan', 'orange', 'purple']
meths = ['eivarx', 'eivar', 'lhs', 'rnd', 'unif']
for mid, m in enumerate(meths):   
    p = np.array([r['Prediction Error'][ninit:nmax] for r in result if r['method'] == m])
    meanerror = np.mean(p, axis=0)
    sderror = np.std(p, axis=0)
    plt.plot(meanerror, label=m, c=cols[mid])
    plt.fill_between(np.arange(0, nmax-ninit), meanerror-1.96*sderror/np.sqrt(seeds), meanerror+1.96*sderror/np.sqrt(seeds), color=cols[mid], alpha=0.1)
plt.legend(bbox_to_anchor=(1.04, -0.1), ncol=len(meths))  
plt.ylabel('Prediction Error')
plt.yscale('log')
plt.show()


    
for mid, m in enumerate(meths):   
    p = np.array([r['Posterior Error'][ninit:nmax] for r in result if r['method'] == m])
    meanerror = np.mean(p, axis=0)
    sderror = np.std(p, axis=0)
    plt.plot(np.mean(p, axis=0), label=m, c=cols[mid])
    plt.fill_between(np.arange(0, nmax-ninit), meanerror-1.96*sderror/np.sqrt(seeds), meanerror+1.96*sderror/np.sqrt(seeds), color=cols[mid], alpha=0.1)
plt.legend(bbox_to_anchor=(1.04, -0.1), ncol=len(meths))  
plt.ylabel('Posterior Error')
plt.yscale('log')
plt.show()