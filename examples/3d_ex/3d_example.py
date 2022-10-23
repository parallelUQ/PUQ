import seaborn as sns
import pandas as pd
import scipy.stats as sps
from generate_test_data import generate_test_data
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output

class gaussian3d:
    def __init__(self):
        self.data_name   = 'gaussian3d'
        self.thetalimits = np.array([[-4, 4], [-4, 4], [-4, 4]])
        self.obsvar      = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]) 
        self.real_data   = np.array([[0, 0, 0]], dtype='float64')  
        self.out         = [('f', float, (3,))]
        self.d           = 3
        self.p           = 3
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2, theta3):
        a = np.array((theta1, theta2, theta3))[:, None]
        b = np.repeat(0.5*a, 3, axis=1).T
        np.fill_diagonal(b, a)
        f = (a.T@b).flatten()
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the gaussian3d function
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2])
        
        return H_o, persis_info

args        = parse_arguments()
cls_3d      = gaussian3d()
test_data   = generate_test_data(cls_3d)

al_3d = designer(data_cls=cls_3d, 
                 method='SEQCAL', 
                 args={'mini_batch': args.minibatch, 
                       'n_init_thetas': 20,
                       'nworkers': args.nworkers,
                       'AL': args.al_func,
                       'seed_n0': args.seed_n0,
                       'prior': 'uniform',
                       'data_test': test_data,
                       'max_evals': 220})

save_output(al_3d, cls_3d.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = False
if show:
    theta_al = al_3d._info['theta']
    TV       = al_3d._info['TV']
    HD       = al_3d._info['HD']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[20:])), TV[20:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()
