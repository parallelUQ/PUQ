from main_deterministic import runfunction
import numpy as np
from PUQ.prior import prior_dist
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import scipy.stats as sps

class covid19:
    def __init__(self):
        self.data_name   = 'covid19'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        self.truelims    = [[1.9, 3.9], [0.29, 1.4], [3, 5], [3, 5]]
        #self.truelims    = [[2.5, 3.5], [0.5, 0.75], [3.5, 4.5], [3.5, 4.5]]
        #self.truelims    = [[2.5, 3.5], [0.29, 1.4], [3.5, 4.5], [3.5, 4.5]]
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
        self.des = []
        self.nrep        = 1
        self.sigma2      = 1**2
        self.nodata      = True        
        

    def function(self, x, theta1, theta2, theta3, theta4):
        
        hosp_ad = np.log(runfunction(x, [theta1, theta2, theta3, theta4], self.truelims, point=True))
        # f = np.sin(10*x - 5*theta)
        return hosp_ad
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2], H['thetas'][0][3], H['thetas'][0][4])
        
        return H_o, persis_info
    
    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        lm = self.truelims
        hosp_ad, daily_ad_benchmark = runfunction(None, [(2.9 - lm[0][0])/(lm[0][1] - lm[0][0]), 
                                                         (0.66 - lm[1][0])/(lm[1][1] - lm[1][0]), 
                                                         (4 - lm[2][0])/(lm[2][1] - lm[2][0]), 
                                                         (4 - lm[3][0])/(lm[3][1] - lm[3][0])], lm, point=False)
        
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv = np.array([np.log(hosp_ad[int(x*221)])])
                newd['feval'].append(fv)
                newd['isreal'] = 'Yes'
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
    
    def realvar(self, x):
        obsvar = np.zeros(x.shape)
        obsvar[x >= 0] = self.sigma2 
        obsvar = obsvar.ravel()
        return obsvar

    def genobsdata(self, x, sigma2, isbias=False):
        varval = self.realvar(x)
        return self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(varval), 1) 


cls_data = covid19()
cls_data.realdata(np.array([75/221, 100/221, 125/221, 150/221, 175/221])[:, None], 1)
prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
prior_t      = prior_dist(dist='uniform')(a=cls_data.thetalimits[1:5][:, 0], b=cls_data.thetalimits[1:5][:, 1])

xint = cls_data.x*221
f = np.zeros((len(xint), 1))
for j in range(len(xint)):
    f[j, 0] = cls_data.function(xint[j]/221, cls_data.true_theta[0], cls_data.true_theta[1], cls_data.true_theta[2], cls_data.true_theta[3])
plt.plot(xint, f[:, 0])
plt.scatter(cls_data.x*221, cls_data.real_data.flatten())
plt.show()

priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

# # # Create a mesh for test set # # # 
xmesh = (np.arange(0, 222)/221)[17:, None]
sampling = LHS(xlimits=cls_data.thetalimits[1:5], random_state=1)
n_t = 500
thetamesh = sampling(n_t)
n_x = cls_data.x.shape[0]
n_tot = n_t*n_x
    
xt_test = np.zeros((n_tot, 5))
ftest = np.zeros((n_t, n_x))
k = 0
for j in range(n_t):
    hosp_ad, daily_ad_benchmark = runfunction(None, [thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2], thetamesh[j, 3]], cls_data.truelims, point=False)
    for i in range(n_x):
        xt_test[k, :] = np.array([cls_data.x[i, 0], thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2], thetamesh[j, 3]])
        ftest[j, i] = np.log(daily_ad_benchmark[int(cls_data.x[i]*221)])
        k += 1

ptest = np.zeros(n_t)
for j in range(n_t):
    rnd = sps.multivariate_normal(mean=ftest[j, :], cov=np.diag(np.repeat(cls_data.sigma2, len(cls_data.x))))
    ptest[j] = rnd.pdf(cls_data.real_data)

xint = cls_data.x*221 
for i in range(n_t):
    if ptest[i] >= 0.000001:
        plt.plot(xint, ftest[i, :], zorder=1)
plt.scatter(cls_data.x*221, cls_data.real_data.flatten(), zorder=2)
plt.show()
       

import seaborn as sns
import pandas as pd
sns.pairplot(pd.DataFrame(thetamesh[ptest > 0.001, :]))

ytest = np.log(hosp_ad)[17:]
#ytest[ytest == -np.inf] = 0

test_data = {'theta': xt_test, 
             'f': ftest,
             'p': ptest,
             'y': ytest,
             'th': thetamesh,    
             'xmesh': xmesh,
             'p_prior': 1} 
# # # # # # # # # # # # # # # # # # # # # 

from PUQ.design import designer

ninit = 50
nmax = 100

al_ceivar = designer(data_cls=cls_data, 
                       method='SEQCOMPDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'ceivar',
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax})

xt_eivar = al_ceivar._info['theta']
f_eivar = al_ceivar._info['f']
thetamle_eivar = al_ceivar._info['thetamle'][-1]

from PUQ.surrogate import emulator
x_emu = np.arange(0, 1)[:, None]
emu = emulator(x_emu, 
               xt_eivar, 
               f_eivar[None, :], 
               method='PCGPexp')

xtrue_test = np.array([np.concatenate([xc, cls_data.true_theta]) for xc in xmesh])
emupred = emu.predict(x=x_emu, theta=xtrue_test)
fhat = emupred.mean()

plt.plot(fhat.flatten())
plt.plot(np.array(hosp_ad).flatten())
plt.show()

from PUQ.surrogatemethods.PCGPexp import  postpred
pmeanhat, pvarhat = postpred(emu._info, cls_data.x, xt_test, cls_data.real_data, np.diag(np.repeat(cls_data.sigma2, len(cls_data.x))))
    
plt.plot(al_ceivar._info['HD'][ninit:nmax])
plt.yscale("log")
plt.show()


sns.pairplot(pd.DataFrame(xt_eivar[ninit:nmax]))
plt.show()

for i in range(0, 50):
    plt.scatter(xt_eivar[ninit:nmax][i, 0], f_eivar[ninit:nmax][i])
plt.show()

al_ceivarx = designer(data_cls=cls_data, 
                       method='SEQCOMPDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'ceivarx',
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax})

xt_eivarx = al_ceivarx._info['theta']
f_eivarx = al_ceivarx._info['f']
thetamle_eivarx = al_ceivarx._info['thetamle'][-1]
    

from PUQ.surrogate import emulator
x_emu = np.arange(0, 1)[:, None]
emu = emulator(x_emu, 
               xt_eivarx, 
               f_eivarx[None, :], 
               method='PCGPexp')

xtrue_test = np.array([np.concatenate([xc, cls_data.true_theta]) for xc in xmesh])
emupred = emu.predict(x=x_emu, theta=xtrue_test)
fhat = emupred.mean()

plt.plot(fhat.flatten())
plt.plot(ytest.flatten())
plt.show()

plt.plot(al_ceivarx._info['TV'][50:100])
plt.yscale("log")


sns.pairplot(pd.DataFrame(xt_eivarx[50:100]))
plt.show()

for i in range(0, 150):
    plt.scatter(xt_eivarx[50:200][i, 0], f_eivarx[50:200][i])
plt.show()


# LHS 
from smt.sampling_methods import LHS
sampling = LHS(xlimits=cls_data.thetalimits)
xt_LHS = sampling(nmax)
al_LHS = designer(data_cls=cls_data, 
                       method='SEQGIVEN', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'seed_n0': s,
                             'prior': priors,
                             'data_test': test_data,
                             'max_evals': nmax,
                             'theta_torun': xt_LHS,
                             'bias': False})
xt_LHS = al_LHS._info['theta']
f_LHS = al_LHS._info['f']
thetamle_LHS = al_LHS._info['thetamle'][-1]

sns.pairplot(pd.DataFrame(xt_LHS[50:100]))
plt.show()

plt.plot(al_LHS._info['TV'][50:100])
plt.plot(al_ceivarx._info['TV'][50:100])

plt.plot(al_LHS._info['HD'][50:100])
plt.plot(al_ceivar._info['HD'][50:100])