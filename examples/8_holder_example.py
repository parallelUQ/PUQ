import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
import scipy.stats as sps
from PUQ.prior import prior_dist
from test_funcs import holder
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

args = parse_arguments()
cls_data = holder()

# # # Create a mesh for test set # # # 
xpl = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 50)
ypl = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
setattr(cls_data, 'theta', th.T)

ftest = np.zeros(2500)
for tid, t in enumerate(th.T):
    ftest[tid] = cls_data.function(t[0], t[1])
thetatest = th.T 
ptest = np.zeros(thetatest.shape[0])
for i in range(ftest.shape[0]):
    mean = ftest[i] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
    ptest[i] = rnd.pdf(cls_data.real_data)
       
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest,
             'p_prior': 1} 
 # # # # # # # # # # # # # # # # # # # # # 
prior_func = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1])
 # # # # # # # # # # # # # # # # # # # # # 
init_seeds = 1
final_seeds = 10
for s in np.arange(init_seeds, final_seeds):
    al_data = designer(data_cls=cls_data, 
                          method='SEQCAL', 
                          args={'mini_batch': args.minibatch, 
                                'n_init_thetas': 10,
                                'nworkers': args.nworkers,
                                'AL': 'pi', 
                                'seed_n0': int(s),
                                'prior': prior_func,
                                'data_test': test_data,
                                'max_evals': 20,
                                'candsize': args.candsize, 
                                'refsize': args.refsize,
                                'believer': args.believer,
                                'type_init': 'LHS'})
    
    save_output(al_data, cls_data.data_name, args.al_func, args.nworkers, args.minibatch, int(s))
    
    
    show = True
    if show:
        theta_al = al_data._info['theta']
        TV       = al_data._info['TV']
        HD       = al_data._info['HD']
        AE       = al_data._info['AE']
        time     = al_data._info['time']
        
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
        
        plt.scatter(np.arange(len(time[11:])), time[11:])
        plt.yscale('log')
        plt.ylabel('Time')
        plt.show()
        
        fig, ax = plt.subplots()    
        cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap='RdGy')
        ax.scatter(theta_al[10:, 0], theta_al[10:, 1], c='black', marker='+', zorder=2)
        ax.scatter(theta_al[0:10, 0], theta_al[0:10, 1], zorder=2, marker='o', facecolors='none', edgecolors='blue')
        ax.set_xlabel(r'$\theta_1$', fontsize=16)
        ax.set_ylabel(r'$\theta_2$', fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        plt.show()
        