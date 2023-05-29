import seaborn as sns
import pandas as pd
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist

class banana:
    def __init__(self):
        self.data_name   = 'banana'
        self.thetalimits = np.array([[-20, 20], [-10, 5]])
        self.obsvar      = np.array([[10**2, 0], [0, 1]]) 
        self.real_data   = np.array([[1, 3]], dtype='float64')  
        self.out         = [('f', float, (2,))]
        self.p           = 2
        self.d           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2):
        f                = np.array([theta1, theta2 + 0.03*theta1**2])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])

        return H_o, persis_info

args        = parse_arguments()
cls_banana  = banana()

# # # Create a mesh for test set # # # 
xpl = np.linspace(cls_banana.thetalimits[0][0], cls_banana.thetalimits[0][1], 50)
ypl = np.linspace(cls_banana.thetalimits[1][0], cls_banana.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
setattr(cls_banana, 'theta', th.T)

al_banana_test = designer(data_cls=cls_banana, 
                            method='SEQUNIFORM', 
                            args={'mini_batch': 4, 
                                  'n_init_thetas': 10,
                                  'nworkers': 5,
                                  'max_evals': th.shape[1]})

ftest = al_banana_test._info['f']
thetatest = al_banana_test._info['theta']

ptest = np.zeros(thetatest.shape[0])
for i in range(ftest.shape[0]):
    mean = ftest[i, :] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_banana.obsvar)
    ptest[i] = rnd.pdf(cls_banana.real_data)
            
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest,
             'p_prior': 1} 
# # # # # # # # # # # # # # # # # # # # # 
prior_func      = prior_dist(dist='uniform')(a=cls_banana.thetalimits[:, 0], b=cls_banana.thetalimits[:, 1])

al_banana = designer(data_cls=cls_banana, 
                     method='SEQCAL', 
                     args={'mini_batch': args.minibatch, 
                           'n_init_thetas': 10,
                           'nworkers': args.nworkers,
                           'AL': args.al_func,
                           'seed_n0': args.seed_n0,
                           'prior': prior_func,
                           'data_test': test_data,
                           'max_evals': 210,
                           'type_init': None})

save_output(al_banana, cls_banana.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = True
if show:
    theta_al = al_banana._info['theta']
    TV       = al_banana._info['TV']
    HD       = al_banana._info['HD']
    
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
    
