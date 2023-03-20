import seaborn as sns
import pandas as pd
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments


class sinlinear:
    def __init__(self):
        self.data_name   = 'sinlinear'
        self.thetalimits = np.array([[-10, 10]])
        self.obsvar = np.array([[1]], dtype='float64')
        self.real_data = np.array([[0]], dtype='float64') 
        self.out = [('f', float)]
        self.p           = 1
        self.d           = 1
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta):
        f = np.sin(theta) + 0.1*theta
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta = H['thetas'][0]
        H_o['f'] = function(theta)

        return H_o, persis_info
    
args        = parse_arguments()
cls_sinlin  = sinlinear()

# # # Create a mesh for test set # # # 
thetatest    = np.arange(-10, 10, 0.0025)[:, None]
ftest        = cls_sinlin.function(thetatest)
ptest        = sps.norm.pdf(cls_sinlin.real_data, ftest, np.sqrt(cls_sinlin.obsvar))


test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest.T} 
# # # # # # # # # # # # # # # # # # # # # 

al_banana = designer(data_cls=cls_sinlin, 
                     method='SEQCAL', 
                     args={'mini_batch': args.minibatch, 
                           'n_init_thetas': 5,
                           'nworkers': args.nworkers,
                           'AL': args.al_func,
                           'seed_n0': args.seed_n0,
                           'prior': 'uniform',
                           'data_test': test_data,
                           'max_evals': 20})


show = True
if show:
    theta_al = al_banana._info['theta']
    TV       = al_banana._info['TV']
    HD       = al_banana._info['HD']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[5:])), TV[5:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(test_data['theta'][:, 0], test_data['p'].flatten(), color='black')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$p(y|\theta)$')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(test_data['theta'][:, 0], test_data['f'].flatten(), color='black')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\eta(\theta)$')
    plt.show()    