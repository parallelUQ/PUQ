import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test, gather_data
from smt.sampling_methods import LHS
from ctest_funcs import nonlin

def add_result(method_name, phat, s):
    rep = {}
    rep['method'] = method_name
    rep['MAD'] = np.mean(np.abs(phat - ptest))
    rep['repno'] = s
    return rep

s = 0
cls_data = nonlin()
cls_data.realdata(s)

n_t = 100
n_x = 9
n_tot = n_t*n_x
t1 = np.linspace(cls_data.thetalimits[2][0], cls_data.thetalimits[2][1], n_t)

XT = np.zeros((n_tot, 3))
f = np.zeros((n_tot))
thetamesh = np.zeros((n_t, 1))
k = 0
for j in range(n_t):
    for i in range(n_x):
        XT[k, :] = np.array([cls_data.real_x[i, 0], cls_data.real_x[i, 1], t1[j]])
        f[k] = cls_data.function(cls_data.real_x[i, 0], cls_data.real_x[i, 1], t1[j])
        k += 1
    
    thetamesh[j, :] = t1[j]
    
ftest = f.reshape(n_t, n_x)
ptest = np.zeros(n_t)

for j in range(n_t):
    rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
    ptest[j] = rnd.pdf(cls_data.real_data)
    
plt.scatter(thetamesh, ptest)
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
nmax = 100
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
    tacq = xt_acq[ninit:nmax, 2]
    
    plt.scatter(cls_data.real_x[:, 0], cls_data.real_x[:, 1])
    plt.scatter(xacq[:, 0], xacq[:, 1])
    plt.show()
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    
    plt.legend()
    plt.show()
    
    
    plt.hist(tacq[ninit:])
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.show()
