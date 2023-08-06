import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from ptest_funcs import bellcurvesimple
from plots_design import gather_data, fitemu, create_test
from smt.sampling_methods import LHS   
 

# # # # # # # # # # # # # # # # # # # # # 
seeds = 10
n0 = 10
n0seed = 1
for s in range(n0seed, seeds):
    cls_data = bellcurvesimple()
    cls_data.realdata(x=np.array([0])[:, None], seed=s)
    args         = parse_arguments()

    th_vec      = (np.arange(0, 100, 10)/100)[:, None]
    x_vec = (np.arange(-300, 300, 1)/100)[:, None]
    fvec = np.zeros((len(th_vec), len(x_vec)))
    pvec = np.zeros((len(th_vec)))
    for t_id, t in enumerate(th_vec):
        for x_id, x in enumerate(x_vec):
            fvec[t_id, x_id] = cls_data.function(x, t)
        plt.plot(x_vec, fvec[t_id, :]) 

    for d_id, d in enumerate(cls_data.des):
        for r in range(d['rep']):
            plt.scatter(d['x'], d['feval'][r], color='black')
    plt.show()

    # # # Create a mesh for test set # # # 
    xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
    
    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'th': thetamesh[:, None],      
                 'xmesh': xmesh[:, None],
                 'p_prior': 1} 

    prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
    prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
    prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))
    
    priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

    #plt.plot(thetamesh, ptest)
    #plt.show()
    
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQEXPDESBIAS', 
                           args={'mini_batch': 1,
                                 'n_init_thetas': n0,
                                 'nworkers': 2, 
                                 'AL': 'ceivar',
                                 'seed_n0': s, 
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': 60,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': True})
    
    xt_eivar = al_unimodal._info['theta']
    f_eivar = al_unimodal._info['f']


    des = al_unimodal._info['des']
    xdes = [e['x'] for e in des]
    nx_ref = len(xdes)
    fdes = np.array([e['feval'][0] for e in des]).reshape(1, nx_ref)
    xu_des, xcount = np.unique(xdes, return_counts=True)
    repeatth = np.repeat(cls_data.true_theta, len(xu_des))
    for label, x_count, y_count in zip(xcount, xu_des, repeatth):
        plt.annotate(label, xy=(x_count, y_count), xytext=(x_count, y_count))
    plt.scatter(xt_eivar[0:n0, 0], xt_eivar[0:n0, 1], marker='*')
    plt.scatter(xt_eivar[:, 0][n0:], xt_eivar[:, 1][n0:], marker='+')
    plt.axhline(y =cls_data.true_theta, color = 'r')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.legend()
    plt.show()
    
    plt.hist(xt_eivar[:, 1][n0:])
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.xlim(0, 1)
    plt.show()

    plt.hist(xt_eivar[:, 0][n0:])
    plt.xlabel(r'x')
    plt.xlim(-3, 3)
    plt.show()
    
