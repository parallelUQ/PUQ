import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns

class unimodal3:
    def __init__(self):

        self.data_name   = 'unimodal3'
        self.thetalimits = np.array([[-10, 10], [-10, 10]])
        self.obsvar      = np.array([[10]], dtype='float64') 
        self.real_data   = np.array([[-1]], dtype='float64') 
        self.out     = [('f', float)]
        self.p           = 2
        self.d           = 1
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):
        f = -np.cos(theta1) * np.cos(theta2) * np.exp(-((theta1 - np.pi)**2 + (theta2 - np.pi)**2))
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        H_o['f'] = function(H['thetas'][0][0], H['thetas'][0][1])

        return H_o, persis_info

args        = parse_arguments()
cls_unimodal3 = unimodal3()

# # # Create a mesh for test set # # # 
xpl = np.linspace(cls_unimodal3.thetalimits[0][0], cls_unimodal3.thetalimits[0][1], 50)
ypl = np.linspace(cls_unimodal3.thetalimits[1][0], cls_unimodal3.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
setattr(cls_unimodal3, 'theta', th.T)

al_unimodal3_test = designer(data_cls=cls_unimodal3, 
                            method='SEQUNIFORM', 
                            args={'mini_batch': 4, 
                                  'n_init_thetas': 10,
                                  'nworkers': 5,
                                  'max_evals': th.shape[1]})

ftest = al_unimodal3_test._info['f']
thetatest = al_unimodal3_test._info['theta']

ptest = np.zeros(thetatest.shape[0])
for i in range(ftest.shape[0]):
    mean = ftest[i] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_unimodal3.obsvar)
    ptest[i] = rnd.pdf(cls_unimodal3.real_data)

fig, ax = plt.subplots()    
cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap='RdGy')
ax.set_xlabel(r'$\theta_1$', fontsize=16)
ax.set_ylabel(r'$\theta_2$', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
plt.show()
       
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest} 
# # # # # # # # # # # # # # # # # # # # # 

al_unimodal3 = designer(data_cls=cls_unimodal3, 
                      method='SEQCAL', 
                      args={'mini_batch': args.minibatch, 
                            'n_init_thetas': 10,
                            'nworkers': args.nworkers,
                            'AL': args.al_func,
                            'seed_n0': args.seed_n0,
                            'prior': 'uniform',
                            'data_test': test_data,
                            'max_evals': 200,
                            'emutype': 'PC',
                            'candsize': args.candsize,
                            'refsize': args.refsize})#args.refsize})

save_output(al_unimodal3, cls_unimodal3.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = True
if show:
    theta_al = al_unimodal3._info['theta']
    f_al     = al_unimodal3._info['f']
    TV       = al_unimodal3._info['TV']
    HD       = al_unimodal3._info['HD']
    AE       = al_unimodal3._info['AE']
    time     = al_unimodal3._info['time']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[10:])), TV[10:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()

    plt.scatter(np.arange(len(AE[10:])), AE[10:])
    plt.yscale('log')
    plt.ylabel('AE')
    plt.show()
    
    plt.scatter(np.arange(len(time[12:])), time[12:])
    #plt.yscale('log')
    plt.ylabel('Time')
    plt.show()
    
    fig, ax = plt.subplots()    
    cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap='RdGy')
    fig.colorbar(cp, ax=ax)
    ax.scatter(theta_al[10:, 0], theta_al[10:, 1], c='black', marker='+', zorder=2)
    ax.scatter(theta_al[0:10, 0], theta_al[0:10, 1], zorder=2, marker='o', facecolors='none', edgecolors='blue')
    ax.set_xlabel(r'$\theta_1$', fontsize=16)
    ax.set_ylabel(r'$\theta_2$', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    plt.show()
    

