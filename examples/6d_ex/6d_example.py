import seaborn as sns
import pandas as pd
import scipy.stats as sps
from generate_test_data import generate_test_data
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output


class gaussian6d:
    def __init__(self):
        self.data_name   = 'gaussian6d'
        self.thetalimits = np.array([[-4, 4], [-4, 4], [-4, 4], [-4, 4], [-4, 4], [-4, 4]])
        ov               = 0.5
        self.obsvar      = np.diag(np.repeat(ov, 6))
        self.real_data   = np.array([[0, 0, 0, 0, 0, 0]], dtype='float64')  
        self.out         = [('f', float, (6,))]
        self.d           = 6
        self.p           = 6
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
   
        
    def function(self, theta1, theta2, theta3, theta4, theta5, theta6):
        a = np.array((theta1, theta2, theta3, theta4, theta5, theta6))[:, None]
        b = np.repeat(0.5*a, 6, axis=1).T
        np.fill_diagonal(b, a)
        f = (a.T@b).flatten()
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the 6d function
        """
        function = sim_specs['user']['function']
        H_o      = np.zeros(1, dtype=sim_specs['out'])
        H_o['f'] = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2],
                            H['thetas'][0][3], H['thetas'][0][4], H['thetas'][0][5])

        return H_o, persis_info

args        = parse_arguments()
cls_6d      = gaussian6d()
test_data   = generate_test_data(cls_6d)

al_6d = designer(data_cls=cls_6d, 
                 method='SEQCAL', 
                 args={'mini_batch': args.minibatch, 
                       'n_init_thetas': 20,
                       'nworkers': args.nworkers,
                       'AL': args.al_func,
                       'seed_n0': args.seed_n0,
                       'prior': 'uniform',
                       'data_test': test_data,
                       'max_evals': 220})

save_output(al_6d, cls_6d.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = False
if show:
    theta_al = al_6d._info['theta']
    TV       = al_6d._info['TV']
    HD       = al_6d._info['HD']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[20:])), TV[20:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()
