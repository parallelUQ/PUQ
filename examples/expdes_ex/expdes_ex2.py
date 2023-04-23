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
        self.thetalimits = np.array([[-3, 3], [-1, 1]])
        self.obsvar      = np.array([[0.02**2, 0, 0], [0, 0.02**2, 0], [0, 0, 0.02**2]], dtype='float64') #np.array([[0.02**2, 0, 0, 0, 0], [0, 0.02**2, 0, 0, 0], [0, 0, 0.02**2, 0, 0], [0, 0, 0, 0.02**2, 0], [0, 0, 0, 0, 0.02**2]], dtype='float64') 
        
        xspace = np.array([-3,  0, 3])
        thetaspace = 0
  
        fevalno = np.zeros(len(xspace))
        for xid in range(len(xspace)):
            fevalno[xid] = np.exp(-4*(xspace[xid] - thetaspace)**2) + np.random.normal(0, 0.02, 1)
            
        self.real_data   = np.array([fevalno], dtype='float64')
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2

        self.x           = xspace[:, None] # For acquisition
        self.real_x           = xspace[:, None]
    def function(self, x, theta):
        """
        Wraps the unimodal function
        """
        #f = np.zeros(5)
        f = np.exp(-4*(x - theta)**2)#np.sin(10*x - 5*theta)
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
    

th_vec      = (np.arange(-100, 100, 10)/100)[:, None]
x_vec = (np.arange(-300, 300, 1)/100)[:, None]
fvec = np.zeros((len(th_vec), len(x_vec)))
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_unimodal.function(x, t)
    plt.plot(x_vec, fvec[t_id, :]) 
plt.scatter(cls_unimodal.x, cls_unimodal.real_data)
plt.show()

# # # Create a mesh for test set # # # 
tpl = np.linspace(cls_unimodal.thetalimits[1][0], cls_unimodal.thetalimits[1][1], 50)
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
             'p_prior': 1} 

prior_func      = prior_dist(dist='uniform')(a=cls_unimodal.thetalimits[:, 0], b=cls_unimodal.thetalimits[:, 1]) 
# # # # # # # # # # # # # # # # # # # # # 

al_unimodal = designer(data_cls=cls_unimodal, 
                       method='SEQEXPDESA', 
                       args={'mini_batch': 1, #args.minibatch, 
                             'n_init_thetas': 10,
                             'nworkers': 2, #args.nworkers,
                             'AL': 'eivar_exp',
                             'seed_n0': 6, #args.seed_n0, #6
                             'prior': prior_func,
                             'data_test': test_data,
                             'max_evals': 50,
                             'type_init': None})

xth = al_unimodal._info['theta']
plt.plot(al_unimodal._info['TV'][10:])
plt.yscale('log')
plt.show()

plt.scatter(xth[:, 0], xth[:, 1], marker='+')
plt.axhline(y = 0, color = 'r')
plt.xlabel('x')
plt.ylabel(r'$\theta$')
plt.show()

plt.hist(xth[:, 1])
plt.axvline(x = 0, color = 'r')
plt.xlabel(r'$\theta$')
plt.show()

plt.hist(xth[:, 0])
plt.xlabel(r'x')
plt.show()
