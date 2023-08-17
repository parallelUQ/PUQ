import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_LHS, obsdata2, fitemu, create_test, gather_data
from smt.sampling_methods import LHS
from ctest_funcs import multicurve

cls_data = multicurve()
cls_data.realdata(np.array([0.25, 0.25, 0.5, 0.5, 0.75, 0.75])[:, None] ,1)
args         = parse_arguments()

def add_result(method_name, phat, ptest, s):
    rep = {}
    rep['method'] = method_name
    rep['MAD'] = np.mean(np.abs(phat - ptest))
    rep['repno'] = s
    return rep

th_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
x_vec  = (np.arange(0, 100, 1)/100)[:, None]
fvec   = np.zeros((len(th_vec), len(x_vec)))
colors = ['blue', 'orange', 'green', 'red', 'purple', 'blue', 'orange', 'green', 'red', 'purple']
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_data.function(x, t)
    plt.plot(x_vec, fvec[t_id, :], label=r'$\theta=$' + str(t), color=colors[t_id]) 

for d_id, d in enumerate(cls_data.des):
    for r in range(d['rep']):
        plt.scatter(d['x'], d['feval'][r], color='black')
plt.xlabel('x')
plt.legend()
plt.show()

# # # Create a mesh for test set # # # 
xt_test, ftest, ptest, thetamesh, _ = create_test(cls_data)
      
test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh[:, None],    
             'xmesh': 0,
             'p_prior': 1} 


prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}
ninit = 10  
nmax = 50
seeds = 10
result = []
for s in range(seeds):
    # # # # # # # # # # # # # # # # # # # # # 
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQCOMPDES', 
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
                                 'design': False})
    
    xt_eivar = al_unimodal._info['theta']
    f_eivar  = al_unimodal._info['f']

    phat_eivar, pvar_eivar = fitemu(xt_eivar, f_eivar[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar) 
    plot_EIVAR(xt_eivar, cls_data, ninit, xlim1=-3, xlim2=3)
    rep = add_result('eivar', phat_eivar, ptest, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_eivar - ptest)))
    
    plt.plot(thetamesh, phat_eivar, c='blue', linestyle='dashed')
    plt.plot(thetamesh, ptest, c='black')
    plt.fill_between(thetamesh, phat_eivar-np.sqrt(pvar_eivar), phat_eivar+np.sqrt(pvar_eivar), alpha=0.2)
    plt.show()
    
    # LHS 
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_lhs   = sampling(nmax)
    f_lhs    = gather_data(xt_lhs, cls_data)
    phat_lhs, pvar_lhs = fitemu(xt_lhs, f_lhs[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar) 
    
    plot_LHS(xt_lhs, cls_data)
    rep = add_result('lhs', phat_lhs, ptest, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_lhs - ptest)))

    # rnd 
    xt_rnd   = prior_xt.rnd(nmax, seed=s)
    f_rnd    = gather_data(xt_rnd, cls_data)
    phat_rnd, pvar_rnd = fitemu(xt_rnd, f_rnd[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar) 
    
    plot_LHS(xt_rnd, cls_data)
    rep = add_result('rnd', phat_rnd, ptest, s)
    result.append(rep)

    print(np.mean(np.abs(phat_rnd - ptest)))
        
    # Unif
    xuniq = np.unique(cls_data.x)
    t_unif = sps.uniform.rvs(0, 1, size=int(nmax/len(xuniq)))
    xvec = np.tile(xuniq, len(t_unif))
    xt_unif   = np.concatenate((xvec[:, None], np.repeat(t_unif, len(xuniq))[:, None]), axis=1)
    f_unif    = gather_data(xt_unif, cls_data)
    phat_unif, pvar_unif = fitemu(xt_unif, f_unif[:, None], xt_test, thetamesh, cls_data.x, cls_data.real_data, cls_data.obsvar)
    
    plot_LHS(xt_unif, cls_data)
    rep = add_result('unif', phat_unif, ptest, s)
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