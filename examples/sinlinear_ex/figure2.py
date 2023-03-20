import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from PUQ.surrogate import emulator
from PUQ.posterior import posterior

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
emupred_test = emu.predict(x=cls_sinlin.x, theta=thetatest)
emupred_tr   = emu.predict(x=cls_sinlin.x, theta=theta)

posttesthat, postvar = post.predict(thetatest)
posttrhat, posttrvar = post.predict(theta)

# Figure 2 (a)
ft = 20
fig, ax = plt.subplots()
ax.plot(thetatest, ftest, color='black')
ax.plot(thetatest.flatten(), emupred_test.mean().flatten(), linestyle='dashed', color='blue', linewidth=2.5)
plt.fill_between(thetatest.flatten(), 
                 (emupred_test.mean() - np.sqrt(emupred_test.var())).flatten(), 
                 (emupred_test.mean() + np.sqrt(emupred_test.var())).flatten(), alpha=0.3)
ax.scatter(theta.T, emupred_tr.mean(), color='red', s=60)
ax.set_xlabel(r'$\theta$', fontsize=ft)
ax.set_ylabel(r'$\eta(\theta)$', fontsize=ft)
ax.tick_params(axis='both', labelsize=ft)
plt.show()    
    
# Figure 2 (b)
fig, ax = plt.subplots()
ax.plot(thetatest, ptest, color='black')
ax.plot(thetatest, posttesthat, color='blue', linestyle='dashed', linewidth=2.5)
plt.fill_between(thetatest.flatten(), 
                 (posttesthat - np.sqrt(postvar)).flatten(), 
                 (posttesthat + np.sqrt(postvar)).flatten(), alpha=0.3)
ax.scatter(theta.T, posttrhat, color='red', s=60)
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
    