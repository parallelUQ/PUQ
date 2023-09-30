import numpy as np
from PUQ.design import designer
from PUQ.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import plot_EIVAR, obsdata, create_test, add_result, samplingdata
from ptest_funcs import sinfunc
import matplotlib.pyplot as plt

args = parse_arguments()


ninit = 10
nmax = 100
result = []

s = 1

cls_data = sinfunc()
dt = len(cls_data.true_theta)
cls_data.realdata(x=np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None], seed=s)
# Observe
obsdata(cls_data, is_bias=False)
        
prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

# # # Create a mesh for test set # # # 
xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
nmesh = len(xmesh)
cls_data_y = sinfunc()
cls_data_y.realdata(x=xmesh, seed=s)

test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'y': cls_data_y.real_data,
             'th': thetamesh,    
             'xmesh': xmesh,
             'p_prior': 1} 
# # # # # # # # # # # # # # # # # # # # # 
al_ceivarx = designer(data_cls=cls_data, 
                       method='SEQDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'ceivarx',
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax,
                             'theta_torun': None})

xt_eivarx = al_ceivarx._info['theta']
f_eivarx = al_ceivarx._info['f']
thetamle_eivarx = al_ceivarx._info['thetamle'][-1]

save_output(al_ceivarx, cls_data.data_name, 'ceivarx', 2, 1, s)

# plot_EIVAR(xt_eivarx, cls_data, ninit, xlim1=0, xlim2=1)

res = {'method': 'eivarx', 'repno': s, 'Prediction Error': al_ceivarx._info['TV'], 'Posterior Error': al_ceivarx._info['HD']}
result.append(res)
# # # # # # # # # # # # # # # # # # # # # 
al_ceivar = designer(data_cls=cls_data, 
                       method='SEQDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'ceivar',
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax,
                             'theta_torun': None})

xt_eivar = al_ceivar._info['theta']
f_eivar = al_ceivar._info['f']
thetamle_eivar = al_ceivar._info['thetamle'][-1]

save_output(al_ceivar, cls_data.data_name, 'ceivar', 2, 1, s)
# plot_EIVAR(xt_eivar, cls_data, ninit, xlim1=0, xlim2=1)

res = {'method': 'eivar', 'repno': s, 'Prediction Error': al_ceivar._info['TV'], 'Posterior Error': al_ceivar._info['HD']}
result.append(res)

# LHS 
xt_LHS = samplingdata('LHS', nmax-ninit, cls_data, s, prior_xt)
al_LHS = designer(data_cls=cls_data, 
                       method='SEQDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': None,
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax,
                             'theta_torun': xt_LHS})
xt_LHS = al_LHS._info['theta']
f_LHS = al_LHS._info['f']
thetamle_LHS = al_LHS._info['thetamle'][-1]

save_output(al_LHS, cls_data.data_name, 'lhs', 2, 1, s)

res = {'method': 'lhs', 'repno': s, 'Prediction Error': al_LHS._info['TV'], 'Posterior Error': al_LHS._info['HD']}
result.append(res)

# rnd 
xt_RND = samplingdata('Random', nmax-ninit, cls_data, s, prior_xt)
al_RND = designer(data_cls=cls_data, 
                       method='SEQDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': None,
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax,
                             'theta_torun': xt_RND})
xt_RND = al_RND._info['theta']
f_RND = al_RND._info['f']
thetamle_RND = al_RND._info['thetamle'][-1]

save_output(al_RND, cls_data.data_name, 'rnd', 2, 1, s)

res = {'method': 'rnd', 'repno': s, 'Prediction Error': al_RND._info['TV'], 'Posterior Error': al_RND._info['HD']}
result.append(res)



show = True
if show:
    cols = ['blue', 'red', 'cyan', 'orange']
    meths = ['eivarx', 'eivar', 'lhs', 'rnd']
    for mid, m in enumerate(meths):   
        p = np.array([r['Prediction Error'][ninit:nmax] for r in result if r['method'] == m])
        meanerror = np.mean(p, axis=0)
        sderror = np.std(p, axis=0)
        plt.plot(meanerror, label=m, c=cols[mid])
        plt.fill_between(np.arange(0, nmax-ninit), meanerror-1.96*sderror/np.sqrt(seeds), meanerror+1.96*sderror/np.sqrt(seeds), color=cols[mid], alpha=0.1)
    plt.legend(bbox_to_anchor=(1.04, -0.1), ncol=len(meths))  
    plt.ylabel('Prediction Error')
    plt.yscale('log')
    plt.show()
    
        
    for mid, m in enumerate(meths):   
        p = np.array([r['Posterior Error'][ninit:nmax] for r in result if r['method'] == m])
        meanerror = np.mean(p, axis=0)
        sderror = np.std(p, axis=0)
        plt.plot(meanerror, label=m, c=cols[mid])
        plt.fill_between(np.arange(0, nmax-ninit), meanerror-1.96*sderror/np.sqrt(seeds), meanerror+1.96*sderror/np.sqrt(seeds), color=cols[mid], alpha=0.1)
    plt.legend(bbox_to_anchor=(1.04, -0.1), ncol=len(meths))  
    plt.ylabel('Posterior Error')
    plt.yscale('log')
    plt.show()