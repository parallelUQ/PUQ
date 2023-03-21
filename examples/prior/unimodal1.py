import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist

class unimodal:
    def __init__(self):
        self.data_name   = 'unimodal1'
        self.thetalimits = np.array([[-2, 5], [-2, 5]])
        self.obsvar      = np.array([[2]], dtype='float64') 
        self.real_data   = np.array([[0]], dtype='float64') 
        self.out     = [('f', float)]
        self.p           = 2
        self.d           = 1
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):
        f = theta1**2 + theta2**2
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

# 2=(5-3)/1, -5=(-2-3)/1
# (-2, 5)
prior_func      = prior_dist(dist='truncnorm')(a=[-5, -5], b=[2, 2], loc=[3, 3], scale=[1, 1])
#num = prior_func.rnd(10000000)[:, 0]
#plt.hist(num[num < -1])
#plt.show()
like_test       = sps.norm.pdf(cls_unimodal.real_data-ftest, 0, np.sqrt(cls_unimodal.obsvar)) 
prior_test      = prior_func.pdf(thetatest)
post_test       = like_test*prior_test


test_data = {'theta': thetatest, 
             'f': ftest,
             'p': post_test,
             'p_prior': prior_test} 
# # # # # # # # # # # # # # # # # # # # # 

al_unimodal = designer(data_cls=cls_unimodal, 
                       method='SEQCAL', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': 10,
                             'nworkers': 2, 
                             'AL': 'eivar',
                             'seed_n0': 6, 
                             'prior': prior_func,
                             'data_test': test_data,
                             'max_evals': 60,
                             'type_init':'LHS'})

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
    levels = np.array([0.0001, 0.0005, 0.001, 0.003, 0.005, 0.008]) # np.arange(0.0001, 0.005, 0.001)
    cp1 = ax.contour(Xpl, Ypl, like_test.reshape(50, 50)/np.sum(like_test), 20, colors='gray', alpha=0.5)
    cp2 = ax.contour(Xpl, Ypl, prior_test.reshape(50, 50)/np.sum(prior_test), 20, colors='red', alpha=0.5, levels=levels)
    ax.clabel(cp2, inline=1, fontsize=10)
    #ax.clabel(cp2, fontsize=10, inline=1, manual=[(4, 2), (4, 1)])
    cp3 = ax.contour(Xpl, Ypl, post_test.reshape(50, 50)/np.sum(post_test), 20, colors='green', alpha=0.5)
    ax.scatter(theta_al[10:, 0], theta_al[10:, 1], c='black', marker='+', zorder=2)
    ax.scatter(theta_al[0:10, 0], theta_al[0:10, 1], zorder=2, marker='o', facecolors='none', edgecolors='blue')
    lines = [cp1.collections[0], cp2.collections[0], cp3.collections[0]]
    labels = [r'$p(y|\theta)$', r'$p(\theta)$', r'$p(\theta|y)$']
    ax.legend(lines, labels, bbox_to_anchor=(1, -0.2), ncol=3, prop={'size': 16})

    ax.set_xlabel(r'$\theta_1$', fontsize=18)
    ax.set_ylabel(r'$\theta_2$', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    plt.show()
