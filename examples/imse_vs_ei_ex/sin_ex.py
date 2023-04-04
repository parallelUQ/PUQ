import seaborn as sns
import pandas as pd
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.surrogate import emulator
from PUQ.posterior import posterior
from PUQ.prior import prior_dist

class sinlinear:
    def __init__(self):
        self.data_name   = 'sinlinear'
        self.thetalimits = np.array([[-10, 10]])
        self.obsvar = np.array([[1]], dtype='float64')
        self.real_data = np.array([[-5]], dtype='float64') 
        self.out = [('f', float)]
        self.p           = 1
        self.d           = 1
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta):
        f = np.sin(theta) + 0.5*theta
        #f = np.exp(-theta)
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
xpl = np.linspace(cls_sinlin.thetalimits[0][0], cls_sinlin.thetalimits[0][1], 50)
th = xpl[:, None]
setattr(cls_sinlin, 'theta', xpl[:, None])

al_unimodal_test = designer(data_cls=cls_sinlin, 
                            method='SEQUNIFORM', 
                            args={'mini_batch': 4, 
                                  'n_init_thetas': 10,
                                  'nworkers': 5,
                                  'max_evals': th.shape[0]})

ftest = al_unimodal_test._info['f']
thetatest = al_unimodal_test._info['theta'][:, None]


like_test       = sps.norm.pdf(cls_sinlin.real_data-ftest, 0, np.sqrt(cls_sinlin.obsvar)) 
prior_func      = prior_dist(dist='uniform')(a=[-10], b=[10])
prior_test      = prior_func.pdf(theta=thetatest)
post_test       = like_test*prior_test


test_data = {'theta': thetatest, 
             'f': ftest,
             'p': post_test,
             'p_prior': prior_test} 

acq_name = 'ei'
seed_no = 10
al_banana = designer(data_cls=cls_sinlin, 
                     method='SEQCALEMU', 
                     args={'mini_batch': args.minibatch, 
                           'n_init_thetas': 5,
                           'nworkers': args.nworkers,
                           'AL': acq_name, # acq_name, #args.al_func,
                           'seed_n0': seed_no, #args.seed_n0,
                           'prior': prior_func,
                           'data_test': test_data,
                           'max_evals': 15,
                           'type_init':'LHS'})

save_output(al_banana, cls_sinlin.data_name, acq_name, args.nworkers, args.minibatch, seed_no)


show = True
if show:
    theta_al = al_banana._info['theta']
    f_al     = al_banana._info['f']
    TV       = al_banana._info['TV']
    HD       = al_banana._info['HD']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[5:])), TV[5:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()

    from matplotlib.ticker import FormatStrFormatter
    emu = emulator(cls_sinlin.x, 
                   theta_al[:, None], 
                   f_al[None, :], 
                   method='PCGP')

    emu_pred          = emu.predict(x=cls_sinlin.x, theta=thetatest)
    emumean, emuvar   = emu_pred.mean(), emu_pred.var()
    postobj           = posterior(data_cls=cls_sinlin, emulator=emu)
    postmean, postvar = postobj.predict(thetatest)
    postmean          = postmean*prior_test
    postvar           = postvar*(prior_test**2)
    postmean_al, postvar_al = postobj.predict(theta_al[:, None])*prior_func.pdf(theta=theta_al[:, None])
    
    fig, ax = plt.subplots()
    ax.plot(thetatest.flatten(), postmean.flatten(), color='blue', linestyle='dashed')
    ax.fill_between(thetatest.flatten(), 
                     (postmean - 2*np.sqrt(postvar)).flatten(), 
                     (postmean + 2*np.sqrt(postvar)).flatten(), alpha=0.3)
    ax.plot(test_data['theta'].flatten(), test_data['p'].flatten(), color='black')
    ax.scatter(theta_al[5:], postmean_al[5:], color='red', s=60)
    ax.scatter(theta_al[0:5], postmean_al[0:5], facecolors='none', edgecolors='b', s=60)
    ax.set_xlabel(r'$\theta$', fontsize=20)
    ax.set_ylabel(r'$\tilde{p}(\theta|y)$', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.show()
    
    
    fig, ax = plt.subplots()
    ax.plot(thetatest.flatten(), emumean.flatten(), color='blue', linestyle='dashed')
    ax.fill_between(thetatest.flatten(), 
                     (emumean - 2*np.sqrt(emuvar)).flatten(), 
                     (emumean + 2*np.sqrt(emuvar)).flatten(), alpha=0.3)
    ax.plot(test_data['theta'].flatten(), test_data['f'].flatten(), color='black')
    ax.scatter(theta_al[5:], al_banana._info['f'][5:], color='red', s=60)
    ax.scatter(theta_al[0:5], al_banana._info['f'][0:5], facecolors='none', edgecolors='b', s=60)
    ax.set_xlabel(r'$\theta$', fontsize=20)
    ax.set_ylabel(r'$\eta(\theta)$', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    plt.show()    