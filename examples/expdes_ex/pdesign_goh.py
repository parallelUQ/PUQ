import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test_goh, gather_data
from PUQ.prior import prior_dist
from ptest_funcs import gohbostos


seeds = 5
ninit = 30
for s in range(seeds):
    cls_data = gohbostos()
    cls_data.realdata(0)
    args         = parse_arguments()

    xt_test, ftest, ptest, thetamesh, xmesh = create_test_goh(cls_data)

        
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=ptest)
    plt.show()

    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'th': thetamesh,    
                 'xmesh': xmesh,
                 'p_prior': 1} 


    prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
    prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]) 
    prior_t      = prior_dist(dist='uniform')(a=cls_data.thetalimits[2:4, 0], b=cls_data.thetalimits[2:4, 1])

    priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQEXPDESBIAS', 
                           args={'mini_batch': 1,
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivar',
                                 'seed_n0': s, 
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': 100,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': True})
    
    xth = al_unimodal._info['theta']
    plt.plot(al_unimodal._info['TV'][10:])
    plt.yscale('log')
    plt.show()

    des = al_unimodal._info['des']
    xdes = [e['x'] for e in des]
    nx_ref = len(xdes)
    fdes = np.array([e['feval'][0] for e in des]).reshape(1, nx_ref)
    xu_des, xcount = np.unique(xdes, axis=0, return_counts=True)
    plt.scatter(xu_des[:, 0], xu_des[:, 1])
    #for label, x_count, y_count in zip(xcount, xu_des, repeatth):
    #    plt.annotate(label, xy=(x_count, y_count), xytext=(x_count, y_count))
    #plt.scatter(xt_eivar[0:n0, 0], xt_eivar[0:n0, 1], marker='*')
    #plt.scatter(xt_eivar[:, 0][n0:], xt_eivar[:, 1][n0:], marker='+')
    #plt.axhline(y =cls_data.true_theta, color = 'r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    plt.scatter(xth[0:ninit, 0], xth[0:ninit, 1], marker='*')
    plt.scatter(xth[ninit:, 0], xth[ninit:, 1], marker='+')
    #plt.axhline(y = 0.5, color = 'r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    


    plt.scatter(xth[0:ninit, 2], xth[0:ninit, 3], marker='*')
    plt.scatter(xth[ninit:, 2], xth[ninit:, 3], marker='+')
    #plt.axhline(y = 0.5, color = 'r')
    plt.xlabel('theta1')
    plt.ylabel('theta2')
    plt.show()

    cls_data.real_x = np.array(xdes)
    cls_data.real_data = fdes
    n_x = len(xdes)
    n_t = len(thetamesh)
    n_tot = n_t*n_x
    f = np.zeros((n_tot))
    k = 0
    for j in range(n_t):
        for i in range(n_x):
            f[k] = cls_data.function(cls_data.real_x[i, 0], cls_data.real_x[i, 1], thetamesh[j, 0], thetamesh[j, 1])
            k += 1
            
    ftest = f.reshape(n_t, n_x)
    ptest = np.zeros(n_t)
    for j in range(n_t):
        rnd = sps.multivariate_normal(mean=ftest[j, :], cov=np.diag(cls_data.realvar(cls_data.real_x)))
        ptest[j] = rnd.pdf(cls_data.real_data)
        
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=ptest)
    plt.show()