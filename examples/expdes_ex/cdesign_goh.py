import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test_goh, gather_data
from smt.sampling_methods import LHS
from ctest_funcs import gohbostos

def add_result(method_name, phat, s):
    rep = {}
    rep['method'] = method_name
    rep['MAD'] = np.mean(np.abs(phat - ptest))
    rep['repno'] = s
    return rep

s = 2
cls_data = gohbostos()
cls_data.realdata(s)

xt_test, ftest, ptest, thetamesh, _ = create_test_goh(cls_data)

test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh,    
             'xmesh': 0,
             'p_prior': 1} 

prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]) 
prior_t      = prior_dist(dist='uniform')(a=cls_data.thetalimits[2:4, 0], b=cls_data.thetalimits[2:4, 1])

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

seeds = 5
ninit = 30
nmax = 99
result = []
for s in range(seeds):

    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQCOMPDES', 
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
                                 'design': False})
    
    xt_acq = al_unimodal._info['theta']
    f_acq = al_unimodal._info['f']
    xacq = xt_acq[ninit:nmax, 0:2]
    tacq = xt_acq[ninit:nmax, 2:4]
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    plt.show()
    
    plt.scatter(tacq[:, 0], tacq[:, 1])
    plt.show()
    

    phat_eivar, pvar_eivar = fitemu(xt_acq, f_acq[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar)
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=phat_eivar)
    plt.show()
    rep = add_result('eivar', phat_eivar, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_eivar - ptest)))
    
    # lhs
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_lhs   = sampling(nmax)
    f_lhs    = np.zeros(nmax)
    for i in range(nmax):
        f_lhs[i] = cls_data.function(xt_lhs[i, 0], xt_lhs[i, 1], xt_lhs[i, 2], xt_lhs[i, 3])
    
    phat_lhs, pvar_lhs = fitemu(xt_lhs, f_lhs[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar)
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=phat_lhs)
    plt.show()
    
    #rep = add_result('lhs', phat_lhs, s)
    #result.append(rep)
    
    print(np.mean(np.abs(phat_lhs - ptest)))
    
    # rnd 
    xt_rnd   = prior_xt.rnd(nmax, seed=s)
    f_rnd    = np.zeros(nmax)
    for i in range(nmax):
        f_rnd[i] = cls_data.function(xt_rnd[i, 0], xt_rnd[i, 1], xt_rnd[i, 2], xt_rnd[i, 3])
    
    phat_rnd, pvar_rnd = fitemu(xt_rnd, f_rnd[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar)
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=phat_rnd)
    plt.show()
    
    #rep = add_result('rnd', phat_rnd, s)
    #result.append(rep)

    print(np.mean(np.abs(phat_rnd - ptest)))
    
    # Unif
    uniqx = np.unique(cls_data.x, axis=0)
    nf = len(uniqx)
    sampling = LHS(xlimits=cls_data.thetalimits[0:2], random_state=s)
    t_unif   = sampling(int(nmax/nf))
    f_unif   = np.zeros(int(nmax/nf)*nf)
    xt_unif = [np.concatenate((uniqx, np.repeat(t.reshape(1, 2), nf, axis=0)), axis=1) for t in t_unif]
    xt_unif = np.array([m for mesh in xt_unif for m in mesh])
    for k in range(nmax):
        f_unif[k] = cls_data.function(xt_unif[k][0], xt_unif[k][1], xt_unif[k][2], xt_unif[k][3])


    phat_unif, pvar_unif = fitemu(xt_unif, f_unif[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar)
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=phat_unif)
    plt.show()
    rep = add_result('unif', phat_unif, s)
    result.append(rep)

    print(np.mean(np.abs(phat_unif - ptest)))

    
import pandas as pd
import seaborn as sns
df = pd.DataFrame(result)
sns.boxplot(x='method', y='MAD', data=df)