import sys
sys.path.append("/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/COVID19")
sys.path.append("/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ")

from main_deterministic import runfunction
import numpy as np
from PUQ.prior import prior_dist
import matplotlib.pyplot as plt
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
        self.sigma2      = 10
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

true_obs, true_func = runfunction(None, [cls_data.true_theta[0], 
                                         cls_data.true_theta[1], 
                                         cls_data.true_theta[2], 
                                         cls_data.true_theta[3]], 
                                  cls_data.truelims, point=False)
ytest = np.array(true_func)[None, :]


from PUQ.utils import parse_arguments, save_output, read_output
from PUQ.surrogate import emulator

path = r'/Users/ozgesurer/Desktop/des_examples/newPUQcovid25/'
w = 2
b = 1
#i = 2
example_name = 'covid19'
out = 'covid19'
metric = 'TV'
method = 'lhs'

for i in range(0, 5):
    design_saved = read_output(path + out + '/', example_name, method, w, b, i)
    
    xt = design_saved._info['theta']
    f = design_saved._info['f']
    x_emu = np.arange(0, 1)[:, None ]
    emu = emulator(x_emu, 
                   xt, 
                   f[None, :], 
                   method='PCGPexp')
    
    xmesh = (np.arange(0, 189)/188)[0:, None]
    xtrue_test = [np.concatenate([xc.reshape(1, 1), np.array(cls_data.true_theta).reshape(1, 4)], axis=1) for xc in xmesh]
    xtrue_test = np.array([m for mesh in xtrue_test for m in mesh])
    predobj = emu.predict(x=x_emu, theta=xtrue_test)
    fmeanhat, fvarhat = predobj.mean(), predobj.var()
        
    import matplotlib.pyplot as plt
    plt.plot(fmeanhat.flatten(), color='b')
    plt.plot(ytest.flatten(), color='r')
    for j in range(len(des_index)):
        plt.axvline(x=des_index[j])
    plt.show()