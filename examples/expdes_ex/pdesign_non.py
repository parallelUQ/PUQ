import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test, gather_data
from smt.sampling_methods import LHS
from ptest_funcs import nonlin
from plots_design import predmle
    
def add_result(method_name, phat, s, ptest):
    rep = {}
    rep['method'] = method_name
    rep['MAD'] = np.mean(np.abs(phat - ptest))
    rep['repno'] = s
    return rep

s = 1
cls_data = nonlin()
cls_data.realdata(s)

cls_data.realvar(cls_data.x)

n_t = 100
n_x = len(cls_data.x)
n_tot = n_t*n_x
thetamesh = np.linspace(cls_data.thetalimits[2][0], cls_data.thetalimits[2][1], n_t).reshape(n_t, 1)
f = np.zeros((n_tot))
k = 0
for j in range(n_t):
    for i in range(n_x):
        f[k] = cls_data.function(cls_data.real_x[i, 0], cls_data.real_x[i, 1], thetamesh[j, 0])
        k += 1
ftest = f.reshape(n_t, n_x)
ptest = np.zeros(n_t)
for j in range(n_t):
    rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
    ptest[j] = rnd.pdf(cls_data.real_data)
    
plt.scatter(thetamesh, ptest)
plt.show()

x1 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 10)
x2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 10)
X1, X2 = np.meshgrid(x1, x2)
XS = np.vstack([X1.ravel(), X2.ravel()]).T

test_data = {'theta': 1, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh,    
             'xmesh': XS,
             'p_prior': 1} 



prior_func      = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 


seeds = 10
ninit = 30
nmax = 100
result1 = []
result2 = []
for s in range(seeds):

    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQEXPDESBIAS', 
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
                                 'design': True})
    
    xt_acq = al_unimodal._info['theta']
    f_acq = al_unimodal._info['f']
    xacq = xt_acq[ninit:nmax, 0:2]
    tacq = xt_acq[ninit:nmax, 2]
    des = al_unimodal._info['des']
    
    xdes = np.array([x['x'] for x in des])
    fdes = np.array([x['feval'] for x in des]).ravel()
    nf = len(xdes)
    
    plt.scatter(cls_data.real_x[:, 0], cls_data.real_x[:, 1])
    plt.scatter(xacq[:, 0], xacq[:, 1])
    plt.show()
    
    unq, cnt = np.unique(xdes, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    
    plt.legend()
    plt.show()
    
    
    plt.hist(tacq[ninit:])
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.show()

    # Construct test data
    testdat = [np.concatenate((xdes, np.repeat(t, nf).reshape(nf, len(t))), axis=1) for t in thetamesh]
    testdat = np.array([m for mesh in testdat for m in mesh])
    ftest       = np.zeros(len(testdat))
    for t_id, t in enumerate(testdat):
        ftest[t_id] = cls_data.function(testdat[t_id, 0], testdat[t_id, 1], testdat[t_id, 2])
    ptest = np.zeros(thetamesh.shape[0])
    ftest = ftest.reshape(len(thetamesh), nf)
    for i in range(ftest.shape[0]):
        mean = ftest[i, :] 
        rnd = sps.multivariate_normal(mean=mean, cov=np.diag(cls_data.realvar(xdes)))
        ptest[i] = rnd.pdf(fdes)

    thmle = np.mean(tacq)
    xtmle = np.concatenate((xdes, np.repeat(thmle, nf).reshape(nf, 1)), axis=1)
    # # #
    
    p_eivar, var_eivar = fitemu(xt_acq, f_acq[:, None], testdat, thetamesh, xdes, fdes[None, :], np.diag(cls_data.realvar(xdes)))
    y_eivar = predmle(xt_acq, f_acq[:, None], xtmle)
    
    print(np.sum((y_eivar - fdes)**2))
    print(np.mean(np.abs(p_eivar - ptest)))
    
    rep = add_result('eivar', p_eivar, s, ptest)
    result1.append(rep)
    rep = add_result('eivar', y_eivar, s, fdes)
    result2.append(rep)
    
    # LHS 
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_lhs   = sampling(100)
    f_lhs    = np.zeros(100)
    for i in range(100):
        f_lhs[i] = cls_data.function(xt_lhs[i][0], xt_lhs[i][1], xt_lhs[i][2])
        
    p_lhs, pvar_lhs = fitemu(xt_lhs, f_lhs[:, None], testdat, thetamesh, xdes, fdes[None, :], np.diag(cls_data.realvar(xdes)))
    y_lhs = predmle(xt_lhs, f_lhs[:, None], xtmle)
    
    print(np.sum((y_lhs - fdes)**2))
    print(np.mean(np.abs(p_lhs - ptest)))

    rep = add_result('lhs', p_lhs, s, ptest)
    result1.append(rep)
    rep = add_result('lhs', y_lhs, s, fdes)
    result2.append(rep)
    
    # rnd 
    xt_rnd   = prior_func.rnd(100, seed=s)
    f_rnd    = np.zeros(100)
    for i in range(100):
        f_rnd[i] = cls_data.function(xt_rnd[i][0], xt_rnd[i][1], xt_rnd[i][2])
    p_rnd, pvar_rnd = fitemu(xt_rnd, f_rnd[:, None], testdat, thetamesh, xdes, fdes[None, :], np.diag(cls_data.realvar(xdes))) 
    y_rnd = predmle(xt_rnd, f_rnd[:, None], xtmle)
    
    print(np.sum((y_rnd - fdes)**2))
    print(np.mean(np.abs(p_rnd - ptest)))
    
    rep = add_result('rnd', p_rnd, s, ptest)
    result1.append(rep)
    
    rep = add_result('rnd', y_rnd, s, fdes)
    result2.append(rep)
    
import pandas as pd
import seaborn as sns
df = pd.DataFrame(result1)
sns.boxplot(x='method', y='MAD', data=df)
plt.show()

df = pd.DataFrame(result2)
sns.boxplot(x='method', y='MAD', data=df)
plt.show()