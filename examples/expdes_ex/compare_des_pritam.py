import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_post, plot_LHS, obsdata, fitemu, create_test_non, gather_data_non, add_result, sampling, samplingdata
from smt.sampling_methods import LHS
from ptest_funcs import pritam
import pandas as pd
import seaborn as sns




seeds = 1
ninit = 30
nmax = 100
result = []
for s in range(seeds):
    
    #x = np.linspace(0, 1, 3)
    #y = np.linspace(0, 1, 3)
    x = np.linspace(0, 1, 2)
    y = np.linspace(0, 1, 2)
    xr = np.array([[xx, yy] for xx in x for yy in y])
    xr = np.concatenate((xr, xr))
    xr = np.array([[0.5, 0.5], [0.5, 0.5]])
    cls_data = pritam()
    cls_data.realdata(xr, seed=s)

    prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
    prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]) 
    prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[2][0]]), b=np.array([cls_data.thetalimits[2][1]]))

    priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

    xt_test, ftest, ptest, thetamesh, xmesh = create_test_non(cls_data)
    cls_data_y = pritam()
    cls_data_y.realdata(x=xmesh, seed=s)
    ytest = cls_data_y.real_data
    
    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'y': ytest,
                 'th': thetamesh,    
                 'xmesh': xmesh,
                 'p_prior': 1} 

    al_ceivarx = designer(data_cls=cls_data, 
                           method='SEQCOMPDES', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivarx',
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'type_init': None})
    
    xt_eivarx = al_ceivarx._info['theta']
    f_eivarx = al_ceivarx._info['f']
    theta_mle = al_ceivarx._info['thetamle'][-1]
    
    res = {'method': 'eivarx', 'repno': s, 'Prediction Error': al_ceivarx._info['TV'], 'Posterior Error': al_ceivarx._info['HD']}
    result.append(res)
    
    xacq = xt_eivarx[ninit:nmax, 0:2]
    tacq = xt_eivarx[ninit:nmax, 2]

    plt.hist(tacq)
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.xlim(0, 1)
    plt.show()
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    plt.show()
    
    
    al_ceivarx_des = designer(data_cls=cls_data, 
                           method='SEQCOMBINEDDES', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivarx',
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax})
    
    xt_eivarx = al_ceivarx_des._info['theta']
    f_eivarx = al_ceivarx_des._info['f']
    theta_mle = al_ceivarx_des._info['thetamle'][-1]
    
    res = {'method': 'eivarx_des', 'repno': s, 'Prediction Error': al_ceivarx_des._info['TV'], 'Posterior Error': al_ceivarx_des._info['HD']}
    result.append(res)
    
    xacq = xt_eivarx[ninit:nmax, 0:2]
    tacq = xt_eivarx[ninit:nmax, 2]

    plt.hist(tacq)
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.xlim(0, 1)
    plt.show()
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    plt.show()
    
cols = ['blue', 'red', 'cyan', 'orange', 'purple']
meths = ['eivarx', 'eivarx_des']
for mid, m in enumerate(meths):   
    p = np.array([r['Prediction Error'][ninit:nmax] for r in result if r['method'] == m])
    print(p)
    meanerror = np.mean(p, axis=0)
    sderror = np.std(p, axis=0)
    plt.plot(meanerror, label=m, c=cols[mid])
    plt.fill_between(np.arange(0, nmax-ninit), meanerror-1.96*sderror/np.sqrt(seeds), meanerror+1.96*sderror/np.sqrt(seeds), color=cols[mid], alpha=0.1)
plt.legend(bbox_to_anchor=(1.04, -0.1), ncol=len(meths))  
plt.ylabel('Prediction Error')
plt.yscale('log')
plt.show()