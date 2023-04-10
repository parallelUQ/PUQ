import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.prior import prior_dist
from smt.sampling_methods import LHS
from PUQ.surrogate import emulator
from PUQ.posterior import posterior
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import eivar_sup


###### ###### ###### ######
###### FIGURE 1 ######
###### ###### ###### ######

# Define the simulation model
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
        Unimodal function
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

cls_unimodal = unimodal()

# Generate test data #
# # # Create a mesh for test set # # # 
xpl = np.linspace(cls_unimodal.thetalimits[0][0], cls_unimodal.thetalimits[0][1], 50)
ypl = np.linspace(cls_unimodal.thetalimits[1][0], cls_unimodal.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
thetatest = th.T
ftest = np.zeros(len(thetatest))
for tid, t in enumerate(thetatest):
    ftest[tid] = cls_unimodal.function(t[0], t[1])
ptest = sps.norm.pdf(cls_unimodal.real_data, ftest, np.sqrt(cls_unimodal.obsvar)) 

test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest,
             'p_prior': 1} 
# # # # # # # # # # # # # # # # # # # # # 

# Define a uniform prior
prior_func      = prior_dist(dist='uniform')(a=cls_unimodal.thetalimits[:, 0], 
                                             b=cls_unimodal.thetalimits[:, 1])

# Run the sequential experimental design with 10 initial sample for 50 acquired points
al_unimodal = designer(data_cls=cls_unimodal, 
                       method='SEQCAL', 
                       args={'mini_batch': 1,
                             'n_init_thetas': 10,
                             'nworkers': 2,
                             'AL': 'eivar',
                             'seed_n0': 1, 
                             'prior': prior_func,
                             'data_test': test_data, 
                             'max_evals': 60,
                             'type_init': None})

# Obtain acquired points
theta_al = al_unimodal._info['theta']

# Figure 1 (b)
fig, ax = plt.subplots()    
cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap='RdGy')
ax.scatter(theta_al[10:, 0], theta_al[10:, 1], c='black', marker='+', zorder=2)
ax.scatter(theta_al[0:10, 0], theta_al[0:10, 1], zorder=2, marker='o', facecolors='none', edgecolors='blue')
ax.set_xlabel(r'$\theta_1$', fontsize=16)
ax.set_ylabel(r'$\theta_2$', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
plt.show()

# Figure 1 (a)
xlimits = np.array([[-4, 4], [-4, 4]])
sampling = LHS(xlimits=xlimits, random_state=2)
x = sampling(50)
fig, ax = plt.subplots()    
cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap='RdGy')
ax.scatter(x[:, 0], x[:, 1], zorder=2, marker='+', c='black')
ax.set_xlabel(r'$\theta_1$', fontsize=16)
ax.set_ylabel(r'$\theta_2$', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
plt.show()


###### ###### ###### ######
###### FIGURE 2 ######
###### ###### ###### ######

# Define the simulation model

class sinlinear:
    def __init__(self):
        self.data_name   = 'sinlinear'
        self.thetalimits = np.array([[-10, 10]])
        self.obsvar = np.array([[1]], dtype='float64')
        self.real_data = np.array([[0]], dtype='float64') 
        self.out = [('f', float)]
        self.p           = 1
        self.d           = 1
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta):
        f = np.sin(theta) + 0.1*theta
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
    
cls_sinlin  = sinlinear()

# Generate training data
theta   = np.array([-10, -8.7, -7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5, 8.7, 10])[:, None]
f       = cls_sinlin.function(theta)
ptr     = sps.norm.pdf(cls_sinlin.real_data, f, np.sqrt(cls_sinlin.obsvar))

# Fit an emulator
emu = emulator(cls_sinlin.x, 
               theta, 
               f, 
               method='PCGP')

post = posterior(data_cls=cls_sinlin, 
                 emulator=emu)

# Generate test data
thetatest    = np.arange(-10, 10, 0.0025)[:, None]
ftest        = cls_sinlin.function(thetatest)
ptest        = sps.norm.pdf(cls_sinlin.real_data, ftest, np.sqrt(cls_sinlin.obsvar))


# Predict via the emulator
testhat = emu.predict(x=cls_sinlin.x, theta=thetatest)
trhat   = emu.predict(x=cls_sinlin.x, theta=theta)

pmeantest, pvartest = post.predict(thetatest)
pmeantr, pvartr = post.predict(theta)

# Figure 2 (a)
ft = 20
fig, ax = plt.subplots()
ax.plot(thetatest, ftest, color='black')
ax.plot(thetatest.flatten(), testhat.mean().flatten(), linestyle='dashed', color='blue', linewidth=2.5)
plt.fill_between(thetatest.flatten(), 
                 (testhat.mean() - np.sqrt(testhat.var())).flatten(), 
                 (testhat.mean() + np.sqrt(testhat.var())).flatten(), alpha=0.3)
ax.scatter(theta.T, trhat.mean(), color='red', s=60)
ax.set_xlabel(r'$\theta$', fontsize=ft)
ax.set_ylabel(r'$\eta(\theta)$', fontsize=ft)
ax.tick_params(axis='both', labelsize=ft)
plt.show()    
    
# Figure 2 (b)
fig, ax = plt.subplots()
ax.plot(thetatest, ptest, color='black')
ax.plot(thetatest, pmeantest, color='blue', linestyle='dashed', linewidth=2.5)
plt.fill_between(thetatest.flatten(), 
                 (pmeantest - np.sqrt(pvartest)).flatten(), 
                 (pmeantest + np.sqrt(pvartest)).flatten(), alpha=0.3)
ax.scatter(theta.T, pmeantr, color='red', s=60)
ax.set_xlabel(r'$\theta$', fontsize=ft)
ax.set_ylabel(r'$p(y|\theta)$', fontsize=ft)
ax.tick_params(axis='both', labelsize=ft)
plt.show()

# Figure 2 (c)
# Fit an emulator for posterior
emu = emulator(cls_sinlin.x, 
               theta, 
               ptr, 
               method='PCGP')

emupred_test    = emu.predict(x=cls_sinlin.x, theta=thetatest)
emumean_test    = emupred_test.mean()
emumean_var     = emupred_test.var()

fig, ax = plt.subplots()
ax.plot(thetatest, ptest, color='black')
ax.plot(thetatest.flatten(), emumean_test.flatten(), linestyle='dashed', color='blue', linewidth=2.5)
plt.fill_between(thetatest.flatten(), 
                 (emumean_test - np.sqrt(emumean_var)).flatten(), 
                 (emumean_test + np.sqrt(emumean_var)).flatten(), alpha=0.3)
ax.scatter(theta.T, ptr, color='red', s=60)
ax.set_xlabel(r'$\theta$', fontsize=ft)
ax.set_ylabel(r'$p(y|\theta)$', fontsize=ft)
ax.tick_params(axis='both', labelsize=ft)
plt.show()



###### ###### ###### ######
###### FIGURE 3 ######
###### ###### ###### ######

# Generate training data
theta   = np.array([-10, -8.7, -7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5, 8.7, 10])[:, None]
f       = cls_sinlin.function(theta)

# Observe three stages
for i in range(3):

    # Fit an emulator
    emu = emulator(cls_sinlin.x, 
                   theta, 
                   f, 
                   method='PCGP')
    
    post = posterior(data_cls=cls_sinlin, 
                     emulator=emu)
    
    # Generate test data
    thetatest    = np.arange(-10, 10, 0.0025)[:, None]
    ftest        = cls_sinlin.function(thetatest)
    ptest        = sps.norm.pdf(cls_sinlin.real_data, ftest, np.sqrt(cls_sinlin.obsvar))
    
    # Predict via the emulator
    emupred_test = emu.predict(x=cls_sinlin.x, theta=thetatest)
    emupred_tr   = emu.predict(x=cls_sinlin.x, theta=theta)
    
    posttesthat, posttestvar = post.predict(thetatest)
    posttrhat, posttrvar = post.predict(theta)
    
    # Candidate list
    clist          = np.arange(-10, 10.01, 0.1)[:, None]
    
    # Obtain eivar for the parameters in the candidate and existing list
    acq   = eivar_sup(clist, theta, thetatest, emu, cls_sinlin) 
    acqtr = eivar_sup(theta, theta, thetatest, emu, cls_sinlin) 

    # Figure 3 / 1st row
    ft = 20
    cand_theta = clist[np.argmin(acq)]
    fig, ax = plt.subplots()
    ax.plot(clist, acq, color='blue')
    ax.set_xlabel(r'$\theta$', fontsize=ft)
    ax.set_ylabel('EIVAR', fontsize=16)
    ax.tick_params(axis='both', labelsize=ft)
    plt.scatter(cand_theta, np.min(acq), color='green', marker='*', s=200)
    plt.scatter(theta, acqtr, color='red', marker='o', s=60)
    plt.show()   
    
    # Figure 3 / 2nd row
    fig, ax = plt.subplots()
    ax.plot(thetatest, ptest, color='black')
    ax.plot(thetatest, posttesthat, color='blue', linestyle='dashed', linewidth=2.5)
    plt.fill_between(thetatest.flatten(), 
                     (posttesthat - np.sqrt(posttestvar)).flatten(), 
                     (posttesthat + np.sqrt(posttestvar)).flatten(), alpha=0.3)
    
    postcand, postcandvar = post.predict(cand_theta.reshape(1, 1))
    plt.scatter(cand_theta, postcand, color='green', marker='*', s=200)
    ax.scatter(theta.T, posttrhat, color='red', s=60)
    ax.set_xlabel(r'$\theta$', fontsize=ft)
    ax.set_ylabel(r'$p(y|\theta)$', fontsize=ft)
    ax.tick_params(axis='both', labelsize=ft)
    plt.show()
    
    # Include new parameter and feval onto the existing data
    theta = np.concatenate((theta, cand_theta.reshape(1, 1)))
    f = np.concatenate((f, cls_sinlin.function(cand_theta.reshape(1, 1))))