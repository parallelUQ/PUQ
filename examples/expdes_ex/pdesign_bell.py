import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from ptest_funcs import bellcurve
from smt.sampling_methods import LHS
from plots_design import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test, gather_data

def add_result(method_name, phat, ptest, s):
    rep = {}
    rep['method'] = method_name
    rep['MAD'] = np.mean(np.abs(phat - ptest))
    rep['repno'] = s
    return rep

cls_data = bellcurve()
cls_data.realdata(x=np.array([0, 0.25, 0.5, 0.75, 1])[:, None], seed=0)
args         = parse_arguments()

th_vec = (np.arange(0, 100, 10)/100)[:, None]
x_vec = (np.arange(0, 100, 1)/100)[:, None]
fvec = np.zeros((len(th_vec), len(x_vec)))
pvec = np.zeros((len(th_vec)))
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_data.function(x, t)
    plt.plot(x_vec, fvec[t_id, :]) 

for d_id, d in enumerate(cls_data.des):
    for r in range(d['rep']):
        plt.scatter(d['x'], d['feval'][r], color='black')
plt.show()


xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
            
test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh[:, None],      
             'xmesh': xmesh[:, None],
             'p_prior': 1} 

prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}
#plt.plot(thetamesh, ptest)
#plt.show()
# # # # # # # # # # # # # # # # # # # # # 
seeds = 5
n0 = 20
nmax = 70
result = []
for s in range(seeds):
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQCOMBINEDDES', 
                           args={'mini_batch': 1,
                                 'n_init_thetas': n0,
                                 'nworkers': 2, 
                                 'AL': 'ceivar',
                                 'seed_n0': s, 
                                 'prior': priors,
                                 'data_test': test_data,
                                 'max_evals': nmax,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': True})
    
    
    xt_eivar = al_unimodal._info['theta']
    f_eivar = al_unimodal._info['f']

    des = al_unimodal._info['des']
    xdes = np.array([e['x'] for e in des])
    fdes = np.array([e['feval'][0] for e in des]).T
    xu_des, xcount = np.unique(xdes, return_counts=True)
    repeatth = np.repeat(cls_data.true_theta, len(xu_des))
    for label, x_count, y_count in zip(xcount, xu_des, repeatth):
        plt.annotate(label, xy=(x_count, y_count), xytext=(x_count, y_count))
    plt.scatter(xt_eivar[0:n0, 0], xt_eivar[0:n0, 1], marker='*')
    plt.scatter(xt_eivar[:, 0][n0:], xt_eivar[:, 1][n0:], marker='+')
    plt.axhline(y =cls_data.true_theta, color = 'r')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()
    
    plt.hist(xt_eivar[:, 1][n0:])
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.show()

    plt.hist(xt_eivar[:, 0][n0:])
    plt.xlabel(r'x')
    plt.xlim(0, 1)
    plt.show()
    
    # Create updated test data
    obsvardes = np.diag(cls_data.realvar(xdes))
    n_x = len(xdes)
    n_t = len(thetamesh)
    xt_des = [np.concatenate([xdes, np.repeat(th, n_x).reshape((n_x, 1))], axis=1) for th in thetamesh]
    xt_des = np.array([m for mesh in xt_des for m in mesh])
    ft = gather_data(xt_des, cls_data)
    ft = ft.reshape(n_t, n_x)
    ptest = np.zeros(n_t)
    for j in range(n_t):
        rnd = sps.multivariate_normal(mean=ft[j, :], cov=obsvardes)
        ptest[j] = rnd.pdf(fdes)
    
    # EIVAR
    phat_eivar, pvar_eivar = fitemu(xt_eivar, f_eivar[:, None], xt_des, thetamesh, xdes, fdes, obsvardes) 
    rep = add_result('eivar', phat_eivar, ptest, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_eivar - ptest)))

    
    # LHS 
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_lhs   = sampling(nmax)
    f_lhs    = gather_data(xt_lhs, cls_data)
    phat_lhs, pvar_lhs = fitemu(xt_lhs, f_lhs[:, None], xt_des, thetamesh, xdes, fdes, obsvardes) 
    
    rep = add_result('lhs', phat_lhs, ptest, s)
    result.append(rep)
    
    print(np.mean(np.abs(phat_lhs - ptest)))

    # rnd 
    xt_rnd   = prior_xt.rnd(nmax, seed=s)
    f_rnd    = gather_data(xt_rnd, cls_data)
    phat_rnd, pvar_rnd = fitemu(xt_rnd, f_rnd[:, None], xt_des, thetamesh, xdes, fdes, obsvardes) 

    rep = add_result('rnd', phat_rnd, ptest, s)
    result.append(rep)

    print(np.mean(np.abs(phat_rnd - ptest)))

    # Unif
    xuniq = np.unique(xdes)
    t_unif = sps.uniform.rvs(0, 1, size=int(nmax/len(xuniq)))
    xvec = np.tile(xuniq, len(t_unif))
    xt_unif   = np.concatenate((xvec[:, None], np.repeat(t_unif, len(xuniq))[:, None]), axis=1)
    f_unif    = gather_data(xt_unif, cls_data)
    phat_unif, pvar_unif = fitemu(xt_unif, f_unif[:, None], xt_des, thetamesh, xdes, fdes, obsvardes)
    
    #plot_LHS(xt_unif, cls_data)
    rep = add_result('unif', phat_unif, ptest, s)
    result.append(rep)

    
    print(np.mean(np.abs(phat_unif - ptest)))

import pandas as pd
import seaborn as sns
df = pd.DataFrame(result)
sns.boxplot(x='method', y='MAD', data=df)

    #from PUQ.surrogate import emulator
    #x_emu = np.arange(0, 1)[:, None ]
    #emu = emulator(x_emu, 
    #               xtheta, 
    #               fevals[:, None], 
    #               method='PCGPexp')


    #xt_test      = np.concatenate((xmesh[:, None], np.repeat(cls_data.true_theta, len(xmesh))[:, None]), axis=1)
    #true_model  = cls_data.function(xt_test[:, 0], xt_test[:, 1])
    #pred        = emu.predict(x=x_emu, theta=xt_test)
    #mean_pred   = pred.mean()
    #var_pred    = np.sqrt(pred.var())
    #plt.plot(xt_test[:, 0], mean_pred.flatten())
    #plt.scatter(xdes, fdes, color='red')
    #plt.plot(xt_test[:, 0], true_model, color='green')
    #plt.fill_between(xt_test[:, 0], mean_pred.flatten() + 2*var_pred.flatten(), mean_pred.flatten() - 2*var_pred.flatten(), alpha=0.5)
    #plt.show()