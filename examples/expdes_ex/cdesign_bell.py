import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test, gather_data
from smt.sampling_methods import LHS
from ctest_funcs import bellcurve

def add_result(method_name, phat, s):
    rep = {}
    rep['method'] = method_name
    rep['MAD'] = np.mean(np.abs(phat - ptest))
    rep['repno'] = s
    return rep

cls_data = bellcurve()
cls_data.realdata(0)
args         = parse_arguments()
    
seeds = 1
result = []
for s in range(seeds):
    
    
    # Observe
    obsdata(cls_data)

    # # # Create a mesh for test set # # # 
    xt_test, ftest, ptest, thetamesh = create_test(cls_data)
          
    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'th': thetamesh[:, None],    
                 'xmesh': 0,
                 'p_prior': 1} 

    prior_func      = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
    # # # # # # # # # # # # # # # # # # # # # 
    
    ninit = 10
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQCOMPDES', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'eivar_exp',
                                 'seed_n0': s,
                                 'prior': prior_func,
                                 'data_test': test_data,
                                 'max_evals': 50,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': False})
    
    xt_eivar = al_unimodal._info['theta']
    f_eivar  = al_unimodal._info['f']

    phat_eivar, pvar_eivar = fitemu(xt_eivar, f_eivar[:, None], xt_test, thetamesh, cls_data) 
    plot_EIVAR(xt_eivar, cls_data, ninit)
    rep = add_result('eivar', phat_eivar, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_eivar - ptest)))
    
    plt.plot(thetamesh, phat_eivar, c='blue', linestyle='dashed')
    plt.plot(thetamesh, ptest, c='black')
    plt.fill_between(thetamesh, phat_eivar-np.sqrt(pvar_eivar), phat_eivar+np.sqrt(pvar_eivar), alpha=0.2)
    plt.show()
    
    # LHS 
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_lhs   = sampling(50)
    f_lhs    = gather_data(xt_lhs, cls_data)
    phat_lhs, pvar_lhs = fitemu(xt_lhs, f_lhs[:, None], xt_test, thetamesh, cls_data) 
    
    plot_LHS(xt_lhs, cls_data)
    rep = add_result('lhs', phat_lhs, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_lhs - ptest)))

    # rnd 
    xt_rnd   = prior_func.rnd(50, seed=s)
    f_rnd    = gather_data(xt_rnd, cls_data)
    phat_rnd, pvar_rnd = fitemu(xt_rnd, f_rnd[:, None], xt_test, thetamesh, cls_data) 
    
    plot_LHS(xt_rnd, cls_data)
    rep = add_result('rnd', phat_rnd, s)
    result.append(rep)

    print(np.mean(np.abs(phat_rnd - ptest)))
        
    # Unif
    t_unif = sps.uniform.rvs(0, 1, size=10)
    xvec = np.tile(cls_data.x.flatten(), len(t_unif))
    xt_unif   = np.concatenate((xvec[:, None], np.repeat(t_unif, len(cls_data.x))[:, None]), axis=1)
    f_unif    = gather_data(xt_unif, cls_data)
    phat_unif, pvar_unif = fitemu(xt_unif, f_unif[:, None], xt_test, thetamesh, cls_data)
    
    plot_LHS(xt_unif, cls_data)
    rep = add_result('unif', phat_unif, s)
    result.append(rep)

    
    print(np.mean(np.abs(phat_unif - ptest)))
    
    plt.plot(thetamesh, phat_unif, c='blue', linestyle='dashed')
    plt.plot(thetamesh, ptest, c='black')
    plt.fill_between(thetamesh, phat_unif-np.sqrt(pvar_unif), phat_unif+np.sqrt(pvar_unif), alpha=0.2)
    plt.show()


import pandas as pd
import seaborn as sns
df = pd.DataFrame(result)
sns.boxplot(x='method', y='MAD', data=df)