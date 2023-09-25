import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import create_test_non, gather_data_non, add_result, samplingdata, plot_des_pri
from smt.sampling_methods import LHS
from ptest_funcs import pritam
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

seeds = 1
ninit = 30
nmax = 72
result = []
for s in range(seeds):
    bias = True

    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)
    xr = np.array([[xx, yy] for xx in x for yy in y])
    xr = np.concatenate((xr, xr))  

    cls_data = pritam()
    cls_data.realdata(xr, seed=s, isbias=bias)

    biastrue = cls_data.bias(cls_data.x[:, 0], cls_data.x[:, 1])
        
    prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
    prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]) 
    prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[2][0]]), b=np.array([cls_data.thetalimits[2][1]]))

    priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

    xt_test, ftest, ptest, thetamesh, xmesh = create_test_non(cls_data, is_bias=bias)
    cls_data_y = pritam()
    cls_data_y.realdata(x=xmesh, seed=s, isbias=bias)
    ytest = cls_data_y.real_data
    
    #plt.plot(cls_data.bias(xmesh[:, 0], xmesh[:, 1]))
    #plt.show()
    
    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'y': ytest,
                 'th': thetamesh,    
                 'xmesh': xmesh,
                 'p_prior': 1} 

    al_ceivarx = designer(data_cls=cls_data, 
                           method='SEQDESBIAS', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivarxbias',
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'bias': bias})
    
    xt_eivarx = al_ceivarx._info['theta']
    f_eivarx = al_ceivarx._info['f']
    
    # plot_des_pri(xt_eivarx, cls_data)
    
    save_output(al_ceivarx, cls_data.data_name, 'ceivarxbias', 2, 1, s)
    
    res = {'method': 'eivarx', 'repno': s, 'Prediction Error': al_ceivarx._info['TV'], 'Posterior Error': al_ceivarx._info['HD']}
    result.append(res)
    

    # # # # # # # # # # # # # # # # # # # # # 
    al_ceivar = designer(data_cls=cls_data, 
                           method='SEQDESBIAS', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivarbias',
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'bias': bias})
    
    xt_eivar = al_ceivar._info['theta']
    f_eivar = al_ceivar._info['f']
    
    # plot_des_pri(xt_eivar, cls_data, ninit, nmax)
    
    save_output(al_ceivar, cls_data.data_name, 'ceivarbias', 2, 1, s)

    res = {'method': 'eivar', 'repno': s, 'Prediction Error': al_ceivar._info['TV'], 'Posterior Error': al_ceivar._info['HD']}
    result.append(res)

    # LHS 
    xt_LHS, f_LHS = samplingdata('LHS', nmax, cls_data, s, prior_xt, non=True)
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
                                 'bias': bias})
    xt_LHS = al_LHS._info['theta']
    f_LHS = al_LHS._info['f']
    
    save_output(al_LHS, cls_data.data_name, 'lhs', 2, 1, s)

    res = {'method': 'lhs', 'repno': s, 'Prediction Error': al_LHS._info['TV'], 'Posterior Error': al_LHS._info['HD']}
    result.append(res)
    
    # rnd 
    xt_RND, f_RND = samplingdata('Random', nmax, cls_data, s, prior_xt, non=True)
    al_RND = designer(data_cls=cls_data, 
                           method='SEQGIVEN', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'theta_torun': xt_RND,
                                 'bias': bias})
    xt_RND = al_RND._info['theta']
    f_RND = al_RND._info['f']
    
    save_output(al_RND, cls_data.data_name, 'rnd', 2, 1, s)
    
    res = {'method': 'rnd', 'repno': s, 'Prediction Error': al_RND._info['TV'], 'Posterior Error': al_RND._info['HD']}
    result.append(res)
    
    # Unif
    xt_UNIF, f_UNIF = samplingdata('Uniform', nmax, cls_data, s, prior_xt, non=True)
    al_UNIF = designer(data_cls=cls_data, 
                           method='SEQGIVEN', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'theta_torun': xt_UNIF,
                                 'bias': bias})
    xt_UNIF = al_UNIF._info['theta']
    f_UNIF = al_UNIF._info['f']
    
    save_output(al_UNIF, cls_data.data_name, 'unif', 2, 1, s)
    
    res = {'method': 'unif', 'repno': s, 'Prediction Error': al_UNIF._info['TV'], 'Posterior Error': al_UNIF._info['HD']}
    result.append(res)
    
show = True
if show:
    cols = ['blue', 'red', 'cyan', 'orange', 'purple']
    meths = ['eivarx', 'lhs', 'rnd',]
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
    
    
    meths = ['eivar', 'eivarx', 'lhs', 'rnd']  
    for mid, m in enumerate(meths):   
        p = np.array([r['Posterior Error'][ninit:nmax] for r in result if r['method'] == m])
        meanerror = np.mean(p, axis=0)
        sderror = np.std(p, axis=0)
        plt.plot(np.mean(p, axis=0), label=m, c=cols[mid])
        #plt.fill_between(np.arange(0, nmax-ninit), meanerror-1.96*sderror/np.sqrt(seeds), meanerror+1.96*sderror/np.sqrt(seeds), color=cols[mid], alpha=0.1)
    plt.legend(bbox_to_anchor=(1.04, -0.1), ncol=len(meths))  
    plt.ylabel('Posterior Error')
    plt.yscale('log')
    plt.show()