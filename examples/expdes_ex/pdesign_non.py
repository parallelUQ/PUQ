import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test_non, gather_data_non
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

x = np.linspace(0, 1, 2)
y = np.linspace(0, 1, 2)
xr = np.array([[xx, yy] for xx in x for yy in y])
        
cls_data.realdata(x=xr, seed=s)
    

        

prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[2][0]]), b=np.array([cls_data.thetalimits[2][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

xt_test, ftest, ptest, thetamesh, xmesh = create_test_non(cls_data)

test_data = {'theta': 1, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh,    
             'xmesh': xmesh,
             'p_prior': 1} 

seeds = 10
ninit = 30
nmax = 100
result1 = []
result2 = []
for s in range(seeds):

    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQCOMBINEDDES', 
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
                                 'design': True})
    
    xt_acq = al_unimodal._info['theta']
    f_acq = al_unimodal._info['f']
    xacq = xt_acq[ninit:nmax, 0:2]
    tacq = xt_acq[ninit:nmax, 2]
    des = al_unimodal._info['des']
    
    xdes = np.array([x['x'] for x in des])
    fdes = np.array([x['feval'] for x in des]).ravel()
    nf = len(xdes)
    
    plt.scatter(cls_data.x[:, 0], cls_data.x[:, 1])
    plt.scatter(xacq[:, 0], xacq[:, 1])
    plt.show()
    
    unq, cnt = np.unique(xdes, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    
    plt.legend()
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
    plt.xlim(0,1)
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
    xt_rnd   = prior_xt.rnd(100, seed=s)
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