import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_post, plot_LHS, obsdata, fitemu, create_test_non, gather_data_non, add_result, sampling
from smt.sampling_methods import LHS
from ptest_funcs import pritam
import pandas as pd
import seaborn as sns

bias = True
x = np.linspace(0, 1, 2)
y = np.linspace(0, 1, 2)
xr = np.array([[xx, yy] for xx in x for yy in y])
xr = np.concatenate((xr, xr))   
s = 10
cls_data = pritam()
cls_data.realdata(xr, seed=s, isbias=bias)

prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[2][0]]), b=np.array([cls_data.thetalimits[2][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

xt_test, ftest, ptest, thetamesh, xmesh = create_test_non(cls_data)
cls_data_y = pritam()
cls_data_y.realdata(x=xmesh, seed=s, isbias=bias)
ytest = cls_data_y.real_data
test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh,    
             'xmesh': xmesh,
             'p_prior': 1,
             'y': ytest} 


a = np.arange(100)/100
b = np.arange(100)/100
X, Y = np.meshgrid(a, b)
Z1 = cls_data.function(X, Y, cls_data.true_theta)    
plt.contour(X, Y, Z1)
plt.show()

Z2 = cls_data.bias(X, Y)    
plt.contour(X, Y, Z1 + Z2)
plt.show()


seeds = 10
ninit = 30
nmax = 60
result = []
for s in range(seeds):
    
    cls_data_y = pritam()
    cls_data_y.realdata(x=xmesh, seed=0)
    ytest = cls_data_y.real_data

    al_ceivarx = designer(data_cls=cls_data, 
                           method='SEQCOMPDESBIAS', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivarxbias',
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'bias': bias})
    
    xt_eivarx = al_ceivarx._info['theta']
    f_eivarx = al_ceivarx._info['f']
    theta_mle = al_ceivarx._info['thetamle'][-1]
    
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
    
    xtrue_test = np.concatenate((xmesh, np.repeat(theta_mle, len(xmesh)).reshape(len(xmesh), theta_mle.shape[1])), axis=1)

    phat_eivarx, pvar_eivarx, yhat_eivarx, yvar_eivarx = fitemu(xt_eivarx, f_eivarx[:, None], xt_test, xtrue_test, thetamesh, cls_data) 
    rep = add_result('eivarx', phat_eivarx, ptest, yhat_eivarx, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh[:, None], phat_eivarx, ptest, pvar_eivarx)
    
    # 
    al_ceivar = designer(data_cls=cls_data, 
                           method='SEQCOMPDESBIAS', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'ceivarbias',
                                 'seed_n0': s,
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'bias': bias})
    
    xt_eivar = al_ceivar._info['theta']
    f_eivar = al_ceivar._info['f']
    
    xacq = xt_eivar[ninit:nmax, 0:2]
    tacq = xt_eivar[ninit:nmax, 2]

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
    
    phat_eivar, pvar_eivar, yhat_eivar, yvar_eivar = fitemu(xt_eivar, f_eivar[:, None], xt_test, xtrue_test, thetamesh, cls_data) 
    rep = add_result('eivar', phat_eivar, ptest, yhat_eivar, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh[:, None], phat_eivar, ptest, pvar_eivar)
    
    # LHS 
    phat_lhs, pvar_lhs, yhat_lhs, yvar_lhs = sampling('LHS', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh, non=True)
    rep = add_result('lhs', phat_lhs, ptest, yhat_lhs, ytest, s)
    result.append(rep)
    
    plot_post(thetamesh[:, None], phat_lhs, ptest, pvar_lhs)

    # rnd 
    phat_rnd, pvar_rnd, yhat_rnd, yvar_rnd = sampling('Random', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh, non=True)
    rep = add_result('rnd', phat_rnd, ptest, yhat_rnd, ytest, s)
    result.append(rep)

    plot_post(thetamesh[:, None], phat_rnd, ptest, pvar_rnd)

    # Unif
    phat_unif, pvar_unif, yhat_unif, yvar_unif = sampling('Uniform', nmax, cls_data, s, prior_xt, xt_test, xtrue_test, thetamesh, non=True)
    rep = add_result('unif', phat_unif, ptest, yhat_unif, ytest, s)
    result.append(rep)

    plot_post(thetamesh[:, None], phat_unif, ptest, pvar_unif)


df = pd.DataFrame(result)
sns.boxplot(x='method', y='Posterior Error', data=df)
plt.show()
sns.boxplot(x='method', y='Prediction Error', data=df)
plt.show()
