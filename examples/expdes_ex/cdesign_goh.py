import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test, gather_data
from smt.sampling_methods import LHS
from ctest_funcs import gohbostos

def add_result(method_name, phat, s):
    rep = {}
    rep['method'] = method_name
    rep['MAD'] = np.mean(np.abs(phat - ptest))
    rep['repno'] = s
    return rep

s = 0
cls_data = gohbostos()
cls_data.realdata(s)

n_t = 20
n_x = 9
n_tot = n_t*n_t*n_x
t1 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], n_t)
t2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], n_t)

T1, T2 = np.meshgrid(t1, t2)
TS = np.vstack([T1.ravel(), T2.ravel()])

XT = np.zeros((n_tot, 4))
f = np.zeros((n_tot))
thetamesh = np.zeros((n_t*n_t, 2))
k = 0
for j in range(n_t*n_t):
    for i in range(n_x):
        XT[k, :] = np.array([cls_data.real_x[i, 0], cls_data.real_x[i, 1], TS[0][j], TS[1][j]])
        f[k] = cls_data.function(cls_data.real_x[i, 0], cls_data.real_x[i, 1], TS[0][j], TS[1][j])
        k += 1
    
    thetamesh[j, :] = np.array([TS[0][j], TS[1][j]])
    
ftest = f.reshape(n_t*n_t, n_x)
ptest = np.zeros(n_t*n_t)

for j in range(n_t*n_t):
    rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
    ptest[j] = rnd.pdf(cls_data.real_data)
    
plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=ptest)
plt.show()

test_data = {'theta': XT, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh,    
             'xmesh': 0,
             'p_prior': 1} 

prior_func      = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 


print(thetamesh[np.argmax(ptest), :])

seeds = 1
ninit = 30
nmax = 135
result = []
for s in range(seeds):

    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQCOMPDES', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'eivar_exp',
                                 'seed_n0': s,
                                 'prior': prior_func,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': False})
    
    xt_acq = al_unimodal._info['theta']
    f_acq = al_unimodal._info['f']
    xacq = xt_acq[ninit:nmax, 0:2]
    tacq = xt_acq[ninit:nmax, 2:4]
    
    plt.scatter(cls_data.real_x[:, 0], cls_data.real_x[:, 1])
    plt.scatter(xacq[:, 0], xacq[:, 1])
    plt.show()
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    
    plt.legend()
    plt.show()
    
    plt.scatter(tacq[:, 0], tacq[:, 1])
    plt.show()
    
    
    phat_eivar, pvar_eivar = fitemu(xt_acq, f_acq[:, None], XT, thetamesh, cls_data)
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
    
    phat_lhs, pvar_lhs = fitemu(xt_lhs, f_lhs[:, None], XT, thetamesh, cls_data)
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=phat_lhs)
    plt.show()
    
    #rep = add_result('lhs', phat_lhs, s)
    #result.append(rep)
    
    print(np.mean(np.abs(phat_lhs - ptest)))
    
    # rnd 
    xt_rnd   = prior_func.rnd(nmax, seed=s)
    f_rnd    = np.zeros(nmax)
    for i in range(nmax):
        f_rnd[i] = cls_data.function(xt_rnd[i, 0], xt_rnd[i, 1], xt_rnd[i, 2], xt_rnd[i, 3])
    
    phat_rnd, pvar_rnd = fitemu(xt_rnd, f_rnd[:, None], XT, thetamesh, cls_data) 
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=phat_rnd)
    plt.show()
    
    #rep = add_result('rnd', phat_rnd, s)
    #result.append(rep)

    print(np.mean(np.abs(phat_rnd - ptest)))
    
    # Unif
    sampling = LHS(xlimits=cls_data.thetalimits[0:2], random_state=s)
    t_unif   = sampling(15)
    f_unif   = np.zeros(nmax)
    xt_unif  = np.zeros((nmax, 4))
    k = 0
    
    x_unif = np.repeat(cls_data.x, 15, axis=0)
    t_unif_rep = np.tile(t_unif, (9,1))
    xt_unif = np.concatenate((x_unif, t_unif_rep), axis=1)
    for k in range(nmax):
        f_unif[k] = cls_data.function(xt_unif[k][0], xt_unif[k][1], xt_unif[k][2], xt_unif[k][3])


    phat_unif, pvar_unif = fitemu(xt_unif, f_unif[:, None], XT, thetamesh, cls_data)
    plt.scatter(thetamesh[:, 0], thetamesh[:, 1], c=phat_unif)
    plt.show()
    
    rep = add_result('unif', phat_unif, s)
    result.append(rep)


    plt.scatter(xt_unif[:, 0], xt_unif[:, 1])
    plt.show()

    print(np.mean(np.abs(phat_unif - ptest)))

    
import pandas as pd
import seaborn as sns
df = pd.DataFrame(result)
sns.boxplot(x='method', y='MAD', data=df)