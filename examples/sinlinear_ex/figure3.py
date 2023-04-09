import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
from PUQ.surrogate import emulator
from PUQ.posterior import posterior
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import get_emuvar, multiple_pdfs, compute_postvar, compute_eivar_fig

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
    
    
    clist          = np.arange(-10, 10.01, 0.1)[:, None]
    def compute_eivar_here(clist, theta, theta_test, emu):
        
        real_x         = cls_sinlin.real_x
        obs            = cls_sinlin.real_data
        obsvar         = cls_sinlin.obsvar
        obsvar3d       = obsvar.reshape(1, 1, 1)
        d_real         = real_x.shape[0]
        x              = cls_sinlin.x
        emupred_test   = emu.predict(x=cls_sinlin.x, theta=theta_test)
        emumean        = emupred_test.mean()
        emuvar, is_cov = get_emuvar(emupred_test)
        emumeanT       = emumean.T
        emuvarT        = emuvar.transpose(1, 0, 2)
        var_obsvar1    = emuvarT + obsvar3d 
        var_obsvar2    = emuvarT + 0.5*obsvar3d 
        diags          = np.diag(obsvar[real_x, real_x.T])
        coef           = (2**d_real)*(np.sqrt(np.pi)**d_real)*np.sqrt(np.prod(diags))
        
        
        # Get the n_ref x d x d x n_cand phi matrix
        emuphi4d      = emu.acquisition(x=x, 
                                        theta1=theta_test, 
                                        theta2=clist)
        acq_func = []
        
        # Pass over all the candidates
        for c_id in range(len(clist)):
            posteivar = compute_eivar_fig(obsvar, 
                                      var_obsvar2[:, real_x, real_x.T],
                                      var_obsvar1[:, real_x, real_x.T], 
                                      emuphi4d[:, real_x, real_x.T, c_id],
                                      emumeanT[:, real_x.flatten()], 
                                      emuvar[real_x, :, real_x.T], 
                                      obs, 
                                      is_cov)
            acq_func.append(posteivar)
    
        return acq_func
    
    acq   = compute_eivar_here(clist, theta, thetatest, emu)
    acqtr = compute_eivar_here(theta, theta, thetatest, emu)
    
    ft = 20
    # Figure 3 (a)/ 2nd row
    cand_theta = clist[np.argmin(acq)]
    fig, ax = plt.subplots()
    ax.plot(clist, acq, color='blue')
    ax.set_xlabel(r'$\theta$', fontsize=ft)
    ax.set_ylabel('EIVAR', fontsize=16)
    ax.tick_params(axis='both', labelsize=ft)
    plt.scatter(cand_theta, np.min(acq), color='green', marker='*', s=200)
    plt.scatter(theta, acqtr, color='red', marker='o', s=60)
    plt.show()   
    
    
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
    
    
    
    theta = np.concatenate((theta, cand_theta.reshape(1, 1)))
    f = np.concatenate((f, cls_sinlin.function(cand_theta.reshape(1, 1))))