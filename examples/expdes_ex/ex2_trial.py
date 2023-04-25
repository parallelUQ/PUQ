import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist

class one_D:
    def __init__(self):
        self.data_name   = 'one_D'
        self.thetalimits = np.array([[-5, 5], [0, 1]])
        self.true_theta  = 0.5
        self.sigma2      = 0.05**2
        self.obsvar      = np.diag(np.repeat(self.sigma2, 7))
        xspace = np.array([-5, -3, -1, 0, 1, 3, 5])
        self.des = [{'x': -5, 'feval':[], 'rep': 5}, 
                    {'x': -3, 'feval':[], 'rep': 5}, 
                    {'x': -1, 'feval':[], 'rep': 5}, 
                    {'x': 0, 'feval':[], 'rep': 5}, 
                    {'x': 1, 'feval':[], 'rep': 5}, 
                    {'x': 3, 'feval':[], 'rep': 5},
                    {'x': 5, 'feval':[], 'rep': 5}]
        nrep = 5
        
        fevalno = np.zeros((len(xspace), nrep))
        for xid, e in enumerate(self.des):
            for r in range(e['rep']):
                fv = np.exp(-1*(e['x'] - self.true_theta)**2) + np.random.normal(0, np.sqrt(self.sigma2), 1)
                e['feval'].append(fv)
                fevalno[xid, r] = fv
        
        mean_feval = np.mean(fevalno, axis=1)    
        self.real_data   = np.array([mean_feval], dtype='float64')
        self.real_data_rep = fevalno

        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2

        self.x           = xspace[:, None] # For acquisition
        self.real_x           = xspace[:, None]
    def function(self, x, theta):
        """
        Wraps the unimodal function
        """
        f = np.exp(-1*(x - theta)**2)
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the simulator
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info
 
args         = parse_arguments()
cls_unimodal = one_D()


xs = np.concatenate([np.repeat(e['x'], e['rep']) for e in cls_unimodal.des])
fs = np.concatenate([e['feval'] for e in cls_unimodal.des])

th_vec      = (np.arange(0, 100, 1)/100)[:, None]
#th_vec      = np.array([0, 0.5, 1])[:, None]
x_vec = (np.arange(-500, 500, 1)/100)[:, None]
fvec = np.zeros((len(th_vec), len(x_vec)))
pvec = np.zeros((len(th_vec)))
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_unimodal.function(x, t)
    plt.plot(x_vec, fvec[t_id, :]) 

for i in range(cls_unimodal.real_data_rep.shape[1]):
    plt.scatter(cls_unimodal.x, cls_unimodal.real_data_rep[:, i])
plt.show()


# # # Create a mesh for test set # # # 
tpl = np.linspace(cls_unimodal.thetalimits[1][0], cls_unimodal.thetalimits[1][1], 10)
xdesign_vec = np.tile(cls_unimodal.x.flatten(), len(tpl))
thetatest   = np.concatenate((xdesign_vec[:, None], np.repeat(tpl, len(cls_unimodal.x))[:, None]), axis=1)
setattr(cls_unimodal, 'theta', thetatest)

al_test = designer(data_cls=cls_unimodal, 
                            method='SEQUNIFORM', 
                            args={'mini_batch': 4, 
                                  'n_init_thetas': 10,
                                  'nworkers': 5,
                                  'max_evals': thetatest.shape[0]})

ftest = al_test._info['f']
thetatest = al_test._info['theta']

ptest = np.zeros(tpl.shape[0])
ftest = ftest.reshape(len(tpl), len(cls_unimodal.x))
for i in range(ftest.shape[0]):
    mean = ftest[i, :] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_unimodal.obsvar)
    ptest[i] = rnd.pdf(cls_unimodal.real_data)
            
plt.plot(tpl, ptest)
plt.show()


ptest = np.zeros((tpl.shape[0], len(cls_unimodal.x)))
ftest = ftest.reshape(len(tpl), len(cls_unimodal.x))
for j in range(ftest.shape[1]):
    for i in range(ftest.shape[0]):

        mean = ftest[i, j] 
        rnd = sps.norm(loc=mean, scale=np.sqrt(cls_unimodal.sigma2))
        ptest[i, j] = rnd.pdf(cls_unimodal.real_data[0][j])
    
    print(np.var(ptest[:, j]))
    plt.plot(tpl, ptest[:, j])
    plt.show()
    
    
ptest = np.zeros((tpl.shape[0], len(cls_unimodal.x), len(cls_unimodal.x)))
ftest = ftest.reshape(len(tpl), len(cls_unimodal.x))
for j in range(ftest.shape[1]):
    for k in range(ftest.shape[1]):
        for i in range(ftest.shape[0]):
            
            mean1 = ftest[i, j] 
            mean2 = ftest[i, k]
            # mean = ftest[i, j] 
            rnd = sps.multivariate_normal(mean=[mean1, mean2], cov=np.diag([cls_unimodal.sigma2, cls_unimodal.sigma2]))
            # rnd = sps.norm(loc=mean, scale=np.sqrt(cls_unimodal.sigma2))
            ptest[i, j, k] = rnd.pdf([cls_unimodal.real_data[0][j], cls_unimodal.real_data[0][k]])
        
        print(str(cls_unimodal.x[j]) + str(cls_unimodal.x[k]))
        print(np.round(np.var(ptest[:, j, k]), 4))
        plt.plot(tpl, ptest[:, j, k])
        plt.title(str(cls_unimodal.x[j]) + str(cls_unimodal.x[k]))
        plt.show()