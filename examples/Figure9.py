from main_deterministic import runfunction
import numpy as np
from PUQ.prior import prior_dist
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import scipy.stats as sps
from PUQ.design import designer
from PUQ.utils import save_output, parse_arguments
import datetime
import matplotlib.dates as mdates

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

# Figure8a

# 2020-02-28
colors = np.repeat('blue', 189)
index = np.arange(0, 189, 15)

# Generate some random date-time data
ft = 15
numdays = 222
base = datetime.datetime(2020, 2, 28, 23, 30) 
date_list = [base + datetime.timedelta(days=x) for x in range(0, numdays) if x >= 33]
# Set the locator
locator = mdates.MonthLocator()  # every month
# Specify the format - %b gives us Jan, Feb...
fmt = mdates.DateFormatter('%b')
plt.scatter(date_list, true_obs, c=colors, label='Observed data')
plt.plot(date_list, true_func, c='red', label='Simulation output')
X = plt.gca().xaxis
X.set_major_locator(locator)
# Specify formatter
X.set_major_formatter(fmt)
plt.ylabel('COVID-19 Hospital Admissions', fontsize=ft)
plt.xticks(fontsize=ft-2)
plt.yticks(fontsize=ft-2)
plt.legend(fontsize=ft-2, ncol=1)
plt.savefig("Figure9a.png", bbox_inches="tight")
plt.show()


fall = []
for j in range(n_t):
    hosp_ad, daily_ad_benchmark = runfunction(None, [thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2], thetamesh[j, 3]], cls_data.truelims, point=False)
    fall.append(daily_ad_benchmark)

date_list = [base + datetime.timedelta(days=x) for x in range(0, 222) if x >= 33]
locator = mdates.MonthLocator()  
fmt = mdates.DateFormatter('%b')
for fa in fall:
    plt.plot(date_list, fa, color='gray', zorder=1)
plt.plot(date_list, ytest.flatten(), color='red', zorder=2)  
X = plt.gca().xaxis
X.set_major_locator(locator)
# Specify formatter
X.set_major_formatter(fmt)
plt.xticks(fontsize=ft-2)
plt.yticks(fontsize=ft-2)
plt.savefig("Figure9b.png", bbox_inches="tight")
plt.show()


# import sys
# sys.path.append("/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/COVID19")
# sys.path.append("/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ")