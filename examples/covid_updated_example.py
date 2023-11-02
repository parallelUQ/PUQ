
import sys
sys.path.append("/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/InterventionsMIP")
sys.path.append("/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ")

from main_deterministic import runfunction
import numpy as np
from PUQ.prior import prior_dist
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import scipy.stats as sps
from PUQ.design import designer
from PUQ.utils import save_output, parse_arguments
from covid_obs import obs_covid19, obs_des
    
class covid19:
    def __init__(self):
        self.data_name   = 'covid19'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        #self.truelims    = [[1.9, 3.9], [0.29, 1.4], [3.9, 4.1], [3.9, 4.1]]
        self.truelims    = [[2.4, 3.4], [0.33, 0.99], [3.9, 4.1], [3.9, 4.1]]
        self.true_theta = [(2.9 - self.truelims[0][0])/(self.truelims[0][1] - self.truelims[0][0]), 
                           (0.66 - self.truelims[1][0])/(self.truelims[1][1] - self.truelims[1][0]), 
                           (4 - self.truelims[2][0])/(self.truelims[2][1] - self.truelims[2][0]), 
                           (4 - self.truelims[3][0])/(self.truelims[3][1] - self.truelims[3][0])]
        
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 5
        self.dx          = 1
        self.x           = None
        self.real_data   = None
        self.sigma2      = 25
        self.nodata      = True        
        

    def function(self, x, theta1, theta2, theta3, theta4):
        hosp_ad = runfunction(x, [theta1, theta2, theta3, theta4], self.truelims, point=True)
        return hosp_ad
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        H_o['f'] = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2], H['thetas'][0][3], H['thetas'][0][4])
        return H_o, persis_info
    
    def realdata(self, x, seed, isbias=False):
        np.random.seed(seed)
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        lm = self.truelims
        hosp_ad, daily_ad_benchmark = runfunction(None, self.true_theta, lm, point=False)

        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = np.array([daily_ad_benchmark[int(np.rint(x*188))]]) + np.random.normal(loc=0.0, scale=np.sqrt(self.sigma2), size=1) 
        self.real_data  = np.array([fevals], dtype='float64')


args = parse_arguments()

cls_data = covid19()

des_index = np.arange(0, 189, 15)[:, None]
#des_index = np.array([0, 30, 60, 90, 120, 150, 180])[:, None]
cls_data.realdata(des_index/188, seed=1)
    
prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=cls_data.thetalimits[1:5][:, 0], b=cls_data.thetalimits[1:5][:, 1])

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}


true_obs, true_func = runfunction(None, [cls_data.true_theta[0], cls_data.true_theta[1], cls_data.true_theta[2], cls_data.true_theta[3]], cls_data.truelims, point=False)
plt.plot(np.arange(0, 189), true_func)
ytest = np.array(true_func)[None, :]
plt.scatter(np.arange(0, 189), ytest)
plt.scatter(des_index.flatten(), cls_data.real_data, color='red')
plt.show()

for xm in cls_data.x:
    hosp_ad = runfunction(xm, [cls_data.true_theta[0], cls_data.true_theta[1], cls_data.true_theta[2], cls_data.true_theta[3]], cls_data.truelims, point=True)
    plt.scatter(int(np.rint(xm*188)), hosp_ad)
plt.scatter(des_index.flatten(), [true_func[di] for di in des_index.flatten()], color='red')
plt.show()
# # # Create a mesh for test set # # # 
n_t = 500
n_x = cls_data.x.shape[0]
n_tot = n_t*n_x
xmesh = (np.arange(0, 189)/188)[0:, None]
sampling = LHS(xlimits=cls_data.thetalimits[1:5], random_state=100)
thetamesh = sampling(n_t)
#thetamesh[0] = np.array(cls_data.true_theta)
xt_test = np.zeros((n_tot, 5))
ftest = np.zeros((n_t, n_x))
k = 0
for j in range(n_t):
    hosp_ad, daily_ad_benchmark = runfunction(None, [thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2], thetamesh[j, 3]], cls_data.truelims, point=False)
    for i in range(n_x):
        xt_test[k, :] = np.array([cls_data.x[i, 0], thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2], thetamesh[j, 3]])
        ftest[j, i] = daily_ad_benchmark[int(np.rint(cls_data.x[i]*188))]
        k += 1

ptest = np.zeros(n_t)
for j in range(n_t):
    rnd = sps.multivariate_normal(mean=ftest[j, :], cov=np.diag(np.repeat(cls_data.sigma2, len(cls_data.x))))
    ptest[j] = rnd.pdf(cls_data.real_data)

test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'y': ytest,
             'th': thetamesh,    
             'xmesh': xmesh,
             'p_prior': 1} 
# # # # # # # # # # # # # # # # # # # # # 
ninit = 50
nmax = 150
result = []

args.seedmin = 0
args.seedmax = 1
for s in np.arange(args.seedmin, args.seedmax):

    s = int(s)

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
    
    res = {'method': 'eivar', 'repno': s, 'Prediction Error': al_ceivar._info['TV'], 'Posterior Error': al_ceivar._info['HD']}
    result.append(res)

    save_output(al_ceivar, cls_data.data_name, 'ceivar', 2, 1, s)

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

    res = {'method': 'eivarx', 'repno': s, 'Prediction Error': al_ceivarx._info['TV'], 'Posterior Error': al_ceivarx._info['HD']}
    result.append(res)
    
    save_output(al_ceivarx, cls_data.data_name, 'ceivarx', 2, 1, s)
    
    # LHS 
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_LHS = sampling(nmax)
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
    
    res = {'method': 'lhs', 'repno': s, 'Prediction Error': al_LHS._info['TV'], 'Posterior Error': al_LHS._info['HD']}
    result.append(res)
    
    save_output(al_LHS, cls_data.data_name, 'lhs', 2, 1, s)
    
    # rnd 
    xt_RND = prior_xt.rnd(nmax, seed=s) 
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
    
    res = {'method': 'rnd', 'repno': s, 'Prediction Error': al_RND._info['TV'], 'Posterior Error': al_RND._info['HD']}
    result.append(res)
    
    save_output(al_RND, cls_data.data_name, 'rnd', 2, 1, s)
    
    print('End of ' + str(s))
    
show = True
if show:
    cols = ['blue', 'red', 'cyan', 'orange']
    meths = ['eivarx', 'eivar', 'lhs', 'rnd']
    for mid, m in enumerate(meths):   
        p = np.array([r['Prediction Error'][ninit:nmax] for r in result if r['method'] == m])
        meanerror = np.mean(p, axis=0)
        sderror = np.std(p, axis=0)
        plt.plot(meanerror, label=m, c=cols[mid])
        plt.fill_between(np.arange(0, nmax-ninit), meanerror-1.96*sderror/np.sqrt(args.seedmax), meanerror+1.96*sderror/np.sqrt(args.seedmax), color=cols[mid], alpha=0.1)
    plt.legend(bbox_to_anchor=(1.04, -0.1), ncol=len(meths))  
    plt.ylabel('Prediction Error')
    plt.yscale('log')
    plt.show()
    
    meths = ['eivarx', 'eivar', 'lhs', 'rnd']  
    for mid, m in enumerate(meths):   
        p = np.array([r['Posterior Error'][ninit:nmax] for r in result if r['method'] == m])
        meanerror = np.mean(p, axis=0)
        sderror = np.std(p, axis=0)
        plt.plot(meanerror, label=m, c=cols[mid])
        plt.fill_between(np.arange(0, nmax-ninit), meanerror-1.96*sderror/np.sqrt(args.seedmax), meanerror+1.96*sderror/np.sqrt(args.seedmax), color=cols[mid], alpha=0.1)
    plt.legend(bbox_to_anchor=(1.04, -0.1), ncol=len(meths))  
    plt.ylabel('Posterior Error')
    plt.yscale('log')
    plt.show()

#import pandas as pd
#import seaborn as sns
#sns.pairplot(pd.DataFrame(xt_eivar[50:100][:, 1:5]))
#plt.show()

#plt.scatter(xt_LHS[50:, 0], np.exp(f_LHS[50:]))
#plt.scatter(cls_data.x, np.exp(cls_data.real_data))
#plt.show()

#true_obs, true_func = runfunction(None, [thetamle_eivarx[0][0], thetamle_eivarx[0][1], thetamle_eivarx[0][2], thetamle_eivarx[0][3]], cls_data.truelims, point=False)
#plt.plot(np.arange(n0_ingore, 222), np.log(true_func[n0_ingore:]))
#plt.scatter(des_index.flatten(), cls_data.real_data)
#plt.show()