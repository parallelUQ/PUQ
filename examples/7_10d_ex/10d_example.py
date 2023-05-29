import seaborn as sns
import pandas as pd
import scipy.stats as sps
from generate_test_data import generate_test_data
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist

class gaussian10d:
    def __init__(self):
        lb = -2
        ub = 2
        self.data_name   = 'gaussian10d'
        self.thetalimits = np.array([[lb, ub], [lb, ub], [lb, ub], [lb, ub], [lb, ub], 
                                     [lb, ub], [lb, ub], [lb, ub], [lb, ub], [lb, ub]])
        ov = 0.25
        self.obsvar     = np.diag(np.repeat(ov, 10))
        self.real_data  = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='float64')  
        self.out    = [('f', float, (10,))]
        self.d           = 10
        self.p           = 10
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10):
        a = np.array((theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10))[:, None]
        b = np.repeat(0.5*a, 10, axis=1).T
        np.fill_diagonal(b, a)
        f = (a.T@b).flatten()
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the 10d function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        H_o['f'] = function(H['thetas'][0][0], H['thetas'][0][1], 
                            H['thetas'][0][2], H['thetas'][0][3], 
                            H['thetas'][0][4], H['thetas'][0][5], 
                            H['thetas'][0][6], H['thetas'][0][7], 
                            H['thetas'][0][8], H['thetas'][0][9])
        
        return H_o, persis_info

args        = parse_arguments()
cls_10d     = gaussian10d()
test_data   = generate_test_data(cls_10d)
test_data['p_prior'] = 1

# # # # # # # # # # # # # # # # # # # # # 
prior_func      = prior_dist(dist='uniform')(a=cls_10d.thetalimits[:, 0], b=cls_10d.thetalimits[:, 1])


al_10d = designer(data_cls=cls_10d, 
                  method='SEQCAL', 
                  args={'mini_batch': args.minibatch, 
                        'n_init_thetas': 30,
                        'nworkers': args.nworkers,
                        'AL': args.al_func,
                        'seed_n0': args.seed_n0,
                        'prior': prior_func,
                        'data_test': test_data,
                        'max_evals': 40,
                        'type_init': None})

save_output(al_10d, cls_10d.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = False
if show:
    theta_al = al_10d._info['theta']
    TV       = al_10d._info['TV']
    HD       = al_10d._info['HD']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[30:])), TV[30:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()
