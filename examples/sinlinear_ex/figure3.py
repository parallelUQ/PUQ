import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from PUQ.surrogate import emulator
from PUQ.posterior import posterior
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import eivar_sup

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