import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output

class unimodal:
    def __init__(self):
        self.data_name   = 'unimodal'
        self.thetalimits = np.array([[-4, 4], [-4, 4]])
        self.obsvar      = np.array([[4]], dtype='float64')
        self.real_data   = np.array([[-6]], dtype='float64')
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2):
        """
        Wraps the unimodal function
        """
        thetas           = np.array([theta1, theta2]).reshape((1, 2))
        S                = np.array([[1, 0.5], [0.5, 1]])
        f                = (thetas @ S) @ thetas.T
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
cls_unimodal = unimodal()

# # # Create a mesh for test set # # # 
xpl = np.linspace(cls_unimodal.thetalimits[0][0], cls_unimodal.thetalimits[0][1], 50)
ypl = np.linspace(cls_unimodal.thetalimits[1][0], cls_unimodal.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
setattr(cls_unimodal, 'theta', th.T)

al_unimodal_test = designer(data_cls=cls_unimodal, 
                            method='SEQUNIFORM', 
                            args={'mini_batch': 4, 
                                  'n_init_thetas': 10,
                                  'nworkers': 5,
                                  'max_evals': th.shape[1]})

ftest = al_unimodal_test._info['f']
thetatest = al_unimodal_test._info['theta']
ptest = sps.norm.pdf(cls_unimodal.real_data-ftest, 0, np.sqrt(cls_unimodal.obsvar)) 

test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest} 
# # # # # # # # # # # # # # # # # # # # # 

al_unimodal = designer(data_cls=cls_unimodal, 
                       method='SEQCAL', 
                       args={'mini_batch': 1, #args.minibatch, 
                             'n_init_thetas': 10,
                             'nworkers': 2, #args.nworkers,
                             'AL': 'eivar',
                             'seed_n0': 6, #args.seed_n0, #6
                             'prior': 'uniform',
                             'data_test': test_data, #test_data,
                             'max_evals': 60})

save_output(al_unimodal, cls_unimodal.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = True
if show:
    theta_al = al_unimodal._info['theta']
    TV       = al_unimodal._info['TV']
    HD       = al_unimodal._info['HD']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[10:])), TV[10:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()
    
    fig, ax = plt.subplots()    
    cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap='RdGy')
    ax.scatter(theta_al[10:, 0], theta_al[10:, 1], c='black', marker='+', zorder=2)
    ax.scatter(theta_al[0:10, 0], theta_al[0:10, 1], zorder=2, marker='o', facecolors='none', edgecolors='blue')
    ax.set_xlabel(r'$\theta_1$', fontsize=16)
    ax.set_ylabel(r'$\theta_2$', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    plt.show()
    
    from smt.sampling_methods import LHS
    lb1 = -4
    ub1 = 4
    xlimits = np.array([[lb1, ub1], [lb1, ub1]])
    sampling = LHS(xlimits=xlimits, random_state=2)
    n = 50
    x = sampling(n)
    fig, ax = plt.subplots()    
    cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap='RdGy')
    ax.scatter(x[:, 0], x[:, 1], zorder=2, marker='+', c='black')
    ax.set_xlabel(r'$\theta_1$', fontsize=16)
    ax.set_ylabel(r'$\theta_2$', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    plt.show()