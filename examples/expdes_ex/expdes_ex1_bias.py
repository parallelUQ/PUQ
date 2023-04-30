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
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = np.pi/5
        self.sigma2      = 0.2**2
        self.obsvar      = np.diag(np.repeat(self.sigma2, 1))
        
        nrep = 1
        xspace = np.array([0.25, 0.5, 0.75])
        xspace = np.array([0.5])
        self.des = [#{'x': 0, 'feval':[], 'rep': nrep}, 
                    {'x': 0.25, 'feval':[], 'rep': nrep}, 
                    {'x': 0.5, 'feval':[], 'rep': nrep},
                    {'x': 0.75, 'feval':[], 'rep': nrep}]
                    #{'x': 1, 'feval':[], 'rep': nrep}]
        self.des = [{'x': 0.5, 'feval':[], 'rep': nrep}]       
        
        fevalno = np.zeros((len(xspace), nrep))
        for xid, e in enumerate(self.des):
            for r in range(e['rep']):
                fv = np.sin(10*xspace[xid] - 5*self.true_theta) + np.random.normal(0, np.sqrt(self.sigma2), 1)
                e['feval'].append(fv)
                fevalno[xid, r] = fv

        
        mean_feval = np.mean(fevalno, axis=1)    
        self.real_data   = np.array([mean_feval], dtype='float64')
        self.real_data_rep = fevalno
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = xspace[:, None] 
        self.real_x      = xspace[:, None]
        
    def function(self, x, theta):
        """
        Wraps the unimodal function
        """
        f = np.sin(10*x - 5*theta)
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
    

th_vec = np.array([np.pi/5, 1])[:, None]
x_vec = (np.arange(0, 100, 1)/100)[:, None]
fvec = np.zeros((len(th_vec), len(x_vec)))
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_unimodal.function(x, t)
    plt.plot(x_vec, fvec[t_id, :]) 
    
for i in range(cls_unimodal.real_data_rep.shape[1]):
    plt.scatter(cls_unimodal.x, cls_unimodal.real_data_rep[:, i])
plt.show()
plt.plot(np.var(fvec, axis=0))
plt.show()


th_vec      = (np.arange(0, 100, 10)/100)[:, None]
x_vec = np.array([ 0.5])[:, None]
fvec = np.zeros((len(th_vec), len(x_vec)))
pvec = np.zeros((len(th_vec)))
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_unimodal.function(x, t)
    pvec[t_id] = sps.multivariate_normal(mean=cls_unimodal.real_data.reshape(-1), cov=cls_unimodal.obsvar).pdf(fvec[t_id, :])
plt.plot(th_vec, pvec)
plt.show()


# # # Create a mesh for test set # # # 
tpl = np.linspace(cls_unimodal.thetalimits[1][0], cls_unimodal.thetalimits[1][1], 50)
xmesh = np.linspace(cls_unimodal.thetalimits[0][0], cls_unimodal.thetalimits[0][1], 50)

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
            
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest,
             'th': tpl[:, None],
             'xmesh': xmesh[:, None],
             'p_prior': 1} 

prior_func      = prior_dist(dist='uniform')(a=cls_unimodal.thetalimits[:, 0], b=cls_unimodal.thetalimits[:, 1])

# # # # # # # # # # # # # # # # # # # # # 


al_unimodal = designer(data_cls=cls_unimodal, 
                       method='SEQEXPDESBIAS', 
                       args={'mini_batch': 1, #args.minibatch, 
                             'n_init_thetas': 20,
                             'nworkers': 2, #args.nworkers,
                             'AL': 'eivar_exp',
                             'seed_n0': 6, #args.seed_n0, #6
                             'prior': prior_func,
                             'data_test': test_data,
                             'max_evals': 100,
                             'type_init': None,
                             'unknown_var': False,
                             'design': True})

xth = al_unimodal._info['theta']
plt.plot(al_unimodal._info['TV'][10:])
plt.yscale('log')
plt.show()

plt.scatter(cls_unimodal.x, np.repeat(np.pi/5, len(cls_unimodal.x)), marker='o', color='black')
plt.scatter(xth[0:20, 0], xth[0:20, 1], marker='*')
plt.scatter(xth[20:, 0], xth[20:, 1], marker='+')
plt.axhline(y = np.pi/5, color = 'r')
plt.xlabel('x')
plt.ylabel(r'$\theta$')
plt.show()


plt.scatter(xth[:, 0], xth[:, 1])
plt.show()

plt.hist(xth[:, 0])
plt.show()

plt.hist(xth[:, 1])
plt.axvline(x = np.pi/5, color = 'r')
plt.show()