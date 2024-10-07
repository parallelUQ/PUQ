
# import sys
# sys.path.append("/home/ac.osurer/newPUQcovid/PUQ")
# sys.path.append("/home/ac.osurer/newPUQcovid/PUQ/PUQ")
# sys.path.append("/home/ac.osurer/newPUQcovid/PUQ/examples/COVID19")

from main_deterministic import runfunction
import numpy as np
from PUQ.prior import prior_dist
from smt.sampling_methods import LHS
import scipy.stats as sps
from PUQ.design import designer
from PUQ.utils import save_output, parse_arguments

class covid19:
    def __init__(self):
        self.data_name   = 'covid19'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
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
cls_data.realdata(des_index/188, seed=1)

prior_xt = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t = prior_dist(dist='uniform')(a=cls_data.thetalimits[1:5][:, 0], b=cls_data.thetalimits[1:5][:, 1])
priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

true_obs, true_func = runfunction(None, [cls_data.true_theta[0], 
                                         cls_data.true_theta[1], 
                                         cls_data.true_theta[2], 
                                         cls_data.true_theta[3]], 
                                  cls_data.truelims, point=False)
ytest = np.array(true_func)[None, :]


# # # Create a mesh for test set # # # 
n_t = 500
n_x = cls_data.x.shape[0]
n_tot = n_t*n_x
xmesh = (np.arange(0, 189)/188)[0:, None]
sampling = LHS(xlimits=cls_data.thetalimits[1:5], random_state=100)
thetamesh = sampling(n_t)

xt_test = np.zeros((n_tot, 5))
ftest = np.zeros((n_t, n_x))
k = 0
for j in range(n_t):
    hosp_ad, daily_ad_benchmark = runfunction(None, [thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2], thetamesh[j, 3]], cls_data.truelims, point=False)
    for i in range(n_x):
        xt_test[k, :] = np.array([cls_data.x[i, 0], thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2], thetamesh[j, 3]])
        ftest[j, i] = daily_ad_benchmark[int(np.rint(cls_data.x[i]*188))]
        k += 1
# # # # # # # # # # # # # # # # # # # # # 
    
ninit = 50
nmax = 200
result = []
args.seedmin = 1
args.seedmax = 2
if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):
    
        s = int(s)
        
        cls_data = covid19()
        des_index = np.arange(0, 189, 15)[:, None]
        cls_data.realdata(des_index/188, seed=s)
        
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
    
        al_maxvar = designer(data_cls=cls_data, 
                         method='SEQDES', 
                         args={'mini_batch': 1, 
                               'n_init_thetas': ninit,
                               'nworkers': 2, 
                               'AL': 'maxvar',
                               'seed_n0': s,
                               'prior': priors,
                               'data_test': test_data,
                               'max_evals': nmax,
                               'theta_torun': None,
                               'is_thetamle': True})

        save_output(al_maxvar, cls_data.data_name, 'maxvar', 2, 1, s)
        
        al_imspe = designer(data_cls=cls_data, 
                                method='SEQDES', 
                                args={'mini_batch': 1, 
                                      'n_init_thetas': ninit,
                                      'nworkers': 2, 
                                      'AL': 'imspe',
                                      'seed_n0': s,
                                      'prior': priors,
                                      'data_test': test_data,
                                      'max_evals': nmax,
                                      'theta_torun': None,
                                      'is_thetamle': False})
        
        save_output(al_imspe, cls_data.data_name, 'imspe', 2, 1, s)
        
        # al_ceivarx = designer(data_cls=cls_data, 
        #                        method='SEQDES', 
        #                        args={'mini_batch': 1, 
        #                              'n_init_thetas': ninit,
        #                              'nworkers': 2, 
        #                              'AL': 'ceivarx',
        #                              'seed_n0': s,
        #                              'prior': priors,
        #                              'data_test': test_data,
        #                              'max_evals': nmax,
        #                              'theta_torun': None,
        #                              'is_thetamle': False})
        
        # xt_eivarx = al_ceivarx._info['theta']
        # f_eivarx = al_ceivarx._info['f']
        # thetamle_eivarx = al_ceivarx._info['thetamle'][-1]
    
        # res = {'method': 'eivarx', 'repno': s, 'Prediction Error': al_ceivarx._info['TV'], 'Posterior Error': al_ceivarx._info['HD']}
        # result.append(res)
        
        # save_output(al_ceivarx, cls_data.data_name, 'ceivarx', 2, 1, s)
        
plt.plot(al_imspe._info['TV'][50:], c='b')
plt.plot(al_maxvar._info['TV'][50:], c='r')
plt.show()

plt.plot(al_imspe._info['HD'][50:], c='b')
plt.plot(al_maxvar._info['HD'][50:], c='r')
plt.show()