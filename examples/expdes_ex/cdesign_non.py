import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test_non, gather_data_non
from smt.sampling_methods import LHS
from ctest_funcs import nonlin

def add_result(method_name, phat, ptest, s):
    rep = {}
    rep['method'] = method_name
    rep['MAD'] = np.mean(np.abs(phat - ptest))
    rep['repno'] = s
    return rep

s = 0
cls_data = nonlin()
cls_data.realdata(s)

prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[2][0]]), b=np.array([cls_data.thetalimits[2][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

xt_test, ftest, ptest, thetamesh, _ = create_test_non(cls_data)

test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh,    
             'xmesh': 0,
             'p_prior': 1} 



seeds = 10
ninit = 10
nmax = 45
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
    
    xt_eivar = al_unimodal._info['theta']
    f_eivar  = al_unimodal._info['f']
    xacq = xt_eivar[ninit:nmax, 0:2]
    tacq = xt_eivar[ninit:nmax, 2]
    
    phat_eivar, pvar_eivar = fitemu(xt_eivar, f_eivar[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar) 
    rep = add_result('eivar', phat_eivar, ptest, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_eivar - ptest)))
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    plt.show()
    
    plt.hist(tacq[ninit:])
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.show()
    
    
    # LHS 
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_lhs   = sampling(nmax)
    f_lhs    = gather_data_non(xt_lhs, cls_data)
    phat_lhs, pvar_lhs = fitemu(xt_lhs, f_lhs[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar) 
    
    rep = add_result('lhs', phat_lhs, ptest, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_lhs - ptest)))
    
    # rnd 
    xt_rnd   = prior_xt.rnd(nmax, seed=s)
    f_rnd    = gather_data_non(xt_rnd, cls_data)
    phat_rnd, pvar_rnd = fitemu(xt_rnd, f_rnd[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar) 
    
    rep = add_result('rnd', phat_rnd, ptest, s)
    result.append(rep)

    print(np.mean(np.abs(phat_rnd - ptest)))
    
    # Unif
    t_unif = sps.uniform.rvs(0, 1, size=5)
    mesh_grid = [np.concatenate([cls_data.x, np.repeat(th, 9).reshape((9, 1))], axis=1) for th in t_unif]
    xt_unif = np.array([m for mesh in mesh_grid for m in mesh])
    f_unif  = gather_data_non(xt_unif, cls_data)
    phat_unif, pvar_unif = fitemu(xt_unif, f_unif[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar)
    
    rep = add_result('unif', phat_unif, ptest, s)
    result.append(rep)

    
    print(np.mean(np.abs(phat_unif - ptest)))
        
import pandas as pd
import seaborn as sns
df = pd.DataFrame(result)
sns.boxplot(x='method', y='MAD', data=df)