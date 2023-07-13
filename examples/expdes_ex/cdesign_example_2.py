import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from PUQ.surrogate import emulator
from emufit import fitemu

class one_D:
    def __init__(self):
        self.data_name   = 'one_D'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = 0.5
        self.sigma2      = 0.2**2
        
        xspace          = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
        dx              = len(xspace)
        nrep            = 2
        self.obsvar     = np.diag(np.repeat(self.sigma2/nrep, dx))
        self.des = []
        self.real_data_rep = np.zeros((dx, nrep))
        for xid, x in enumerate(xspace):
            newd = {}
            newd['x'] = x
            newd['feval'] = []
            newd['rep'] = nrep
            for r in range(nrep):
                fv              = np.exp(-100*(x - self.true_theta)**2) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
                self.real_data_rep[xid, r] = fv
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2

        self.x           = xspace[:, None] # For acquisition
        self.real_x      = xspace[:, None]
        
        self.theta_torun = None
        
    def function(self, x, theta):
        """
        Wraps the unimodal function
        """
        f = np.exp(-100*(x - theta)**2) 
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
cls_unimodal = one_D()


xs = np.concatenate([np.repeat(e['x'], e['rep']) for e in cls_unimodal.des])
fs = np.concatenate([e['feval'] for e in cls_unimodal.des])

th_vec      = [0.3, 0.4, 0.5, 0.6, 0.7]#(np.arange(0, 100, 20)/100)[:, None]
x_vec = (np.arange(0, 100, 1)/100)[:, None]
fvec = np.zeros((len(th_vec), len(x_vec)))
pvec = np.zeros((len(th_vec)))
colors = ['blue', 'orange', 'green', 'red', 'purple']
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_unimodal.function(x, t)
    plt.plot(x_vec, fvec[t_id, :], label=r'$\theta=$' + str(t), color=colors[t_id]) 

for i in range(cls_unimodal.real_data_rep.shape[1]):
    if i == 0:
        plt.scatter(cls_unimodal.x, cls_unimodal.real_data_rep[:, i], color='black', label='Data')
    else:
        plt.scatter(cls_unimodal.x, cls_unimodal.real_data_rep[:, i], color='black')

plt.xlabel('x')
plt.legend()
plt.show()


# # # Create a mesh for test set # # # 
thetamesh   = np.linspace(cls_unimodal.thetalimits[1][0], cls_unimodal.thetalimits[1][1], 100)
xdesign_vec = np.tile(cls_unimodal.x.flatten(), len(thetamesh))
thetatest   = np.concatenate((xdesign_vec[:, None], np.repeat(thetamesh, len(cls_unimodal.x))[:, None]), axis=1)
ftest       = np.zeros(len(thetatest))
for t_id, t in enumerate(thetatest):
    ftest[t_id] = cls_unimodal.function(thetatest[t_id, 0], thetatest[t_id, 1])

ptest = np.zeros(thetamesh.shape[0])
ftest = ftest.reshape(len(thetamesh), len(cls_unimodal.x))
for i in range(ftest.shape[0]):
    mean = ftest[i, :] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_unimodal.obsvar)
    ptest[i] = rnd.pdf(cls_unimodal.real_data)
     
plt.plot(thetamesh, ptest)
plt.show()
      
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh[:, None],    
             'xmesh': 0,
             'p_prior': 1} 

prior_func      = prior_dist(dist='uniform')(a=cls_unimodal.thetalimits[:, 0], b=cls_unimodal.thetalimits[:, 1]) 
# # # # # # # # # # # # # # # # # # # # # 



ninit = 10
al_unimodal = designer(data_cls=cls_unimodal, 
                       method='SEQCOMPDES', 
                       args={'mini_batch': 1, #args.minibatch, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, #args.nworkers,
                             'AL': 'eivar_exp',
                             'seed_n0': 6, #args.seed_n0, #6
                             'prior': prior_func,
                             'data_test': test_data,
                             'max_evals': 51,
                             'type_init': None,
                             'unknown_var': False,
                             'design': False})

xt_acq = al_unimodal._info['theta']
f_acq   = al_unimodal._info['f']
TV_acq = al_unimodal._info['TV']

plt.plot(TV_acq[ninit:])
#plt.yscale('log')
plt.show()

save_output(al_unimodal, cls_unimodal.data_name, 'eivar_exp', args.nworkers, args.minibatch, args.seed_n0)


plt.scatter(cls_unimodal.x, np.repeat(cls_unimodal.true_theta, len(cls_unimodal.x)), marker='o', color='black')
plt.scatter(xt_acq[0:ninit, 0], xt_acq[0:ninit, 1], marker='*', color='blue')
plt.scatter(xt_acq[:, 0][ninit:], xt_acq[:, 1][ninit:], marker='+', color='red')
plt.axhline(y = 0.5, color = 'green')
plt.xlabel('x')
plt.ylabel(r'$\theta$')
plt.show()

plt.hist(xt_acq[:, 1][ninit:])
plt.axvline(x = 0.5, color = 'r')
plt.xlabel(r'$\theta$')
plt.show()

plt.hist(xt_acq[:, 0][ninit:])
plt.xlabel(r'x')
plt.xlim(0, 1)
plt.show()

phat = fitemu(xt_acq[0:50], f_acq[0:50], thetatest, thetamesh, cls_unimodal)
print(np.mean(np.abs(phat - ptest)))
plt.scatter(phat, ptest)
plt.show()

print(TV_acq[-1])


from smt.sampling_methods import LHS
sampling = LHS(xlimits=cls_unimodal.thetalimits, random_state=1)
xt_LHS   = sampling(50)
f_LHS    = np.zeros(len(xt_LHS))
for t_id, t in enumerate(xt_LHS):
    f_LHS[t_id] = cls_unimodal.function(xt_LHS[t_id, 0], xt_LHS[t_id, 1])

phat = fitemu(xt_LHS, f_LHS, thetatest, thetamesh, cls_unimodal)
plt.scatter(cls_unimodal.x, np.repeat(cls_unimodal.true_theta, len(cls_unimodal.x)), marker='o', color='black')
plt.scatter(xt_LHS[:, 0], xt_LHS[:, 1], marker='*', color='blue')
plt.axhline(y = 0.5, color = 'green')
plt.xlabel('x')
plt.ylabel(r'$\theta$')
plt.show()

print(np.mean(np.abs(phat - ptest)))
plt.scatter(phat, ptest)
plt.show()

setattr(cls_unimodal, 'theta_torun', np.concatenate((xt_LHS, np.array([0,0]).reshape(1, 2)), axis=0))
al_LHS = designer(data_cls=cls_unimodal, 
                       method='SEQCOMPDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'eivar_exp',
                             'seed_n0': 6, #args.seed_n0, #6
                             'prior': prior_func,
                             'data_test': test_data,
                             'max_evals': 51,
                             'type_init': 'to_run',
                             'unknown_var': False,
                             'design': False})

save_output(al_LHS, cls_unimodal.data_name, 'LHS', args.nworkers, args.minibatch, args.seed_n0)

print(al_LHS._info['TV'][-1])

t_unif = sps.uniform.rvs(0, 1, size=10)
xvec = np.tile(cls_unimodal.x.flatten(), len(t_unif))
xt_unif   = np.concatenate((xvec[:, None], np.repeat(t_unif, len(cls_unimodal.x))[:, None]), axis=1)
f_unif    = np.zeros(len(xt_unif))
for t_id, t in enumerate(xt_unif):
    f_unif[t_id] = cls_unimodal.function(xt_unif[t_id, 0], xt_unif[t_id, 1])

phat = fitemu(xt_unif, f_unif, thetatest, thetamesh, cls_unimodal)
plt.scatter(cls_unimodal.x, np.repeat(cls_unimodal.true_theta, len(cls_unimodal.x)), marker='o', color='black')
plt.scatter(xt_unif[:, 0], xt_unif[:, 1], marker='*', color='blue')
plt.axhline(y = 0.5, color = 'green')
plt.xlabel('x')
plt.ylabel(r'$\theta$')
plt.show()

print(np.mean(np.abs(phat - ptest)))
plt.scatter(phat, ptest)
plt.show()


setattr(cls_unimodal, 'theta_torun', np.concatenate((xt_unif, np.array([0,0]).reshape(1, 2)), axis=0))
al_unif = designer(data_cls=cls_unimodal, 
                       method='SEQCOMPDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'eivar_exp',
                             'seed_n0': 6, #args.seed_n0, #6
                             'prior': prior_func,
                             'data_test': test_data,
                             'max_evals': 51,
                             'type_init': 'to_run',
                             'unknown_var': False,
                             'design': False})


print(al_unif._info['TV'][-1])