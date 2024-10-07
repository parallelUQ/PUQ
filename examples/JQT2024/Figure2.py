import numpy as np
from PUQ.design import designer
from PUQ.prior import prior_dist
from plots_design import create_test
from ptest_funcs import sinfunc
import matplotlib.pyplot as plt
from PUQ.surrogate import emulator
from Figure2support import ceivarfig, ceivarxfig
from PUQ.designmethods.SEQDESsupport import find_mle


options = [0, 1]
fig, axs = plt.subplots(2, 3, figsize=(15, 7))
for o in options:
    s = 1

    cls_data = sinfunc()
    dt = len(cls_data.true_theta)
    cls_data.realdata(x=np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None], seed=s)

    prior_xt     = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
    prior_x      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[0][0]]), b=np.array([cls_data.thetalimits[0][1]])) 
    prior_t      = prior_dist(dist='uniform')(a=np.array([cls_data.thetalimits[1][0]]), b=np.array([cls_data.thetalimits[1][1]]))

    priors = {'prior': prior_xt, 'priorx': prior_x, 'priort': prior_t}

    # # # Create a mesh for test set # # # 
    xt_test, ftest, ptest, thetamesh, xmesh = create_test(cls_data)
    nmesh = len(xmesh)
    cls_data_y = sinfunc()
    cls_data_y.realdata(x=xmesh, seed=s)
    ytest = cls_data_y.real_data

    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'y': ytest,
                 'th': thetamesh,    
                 'xmesh': xmesh,
                 'p_prior': 1} 
    # # # # # # # # # # # # # # # # # # # # # 
    x_emu = np.arange(0, 1)[:, None ]
    sinit = 5
    ninit = 10
    nmax = 30

    # Create initial data
    xt = prior_xt.rnd(ninit, sinit) 
    f = cls_data.function(xt[:, 0], xt[:, 1])
    
    if o == 0:
        # Acquire new points
        for i in range(nmax-ninit):
            emu = emulator(x_emu, 
                           xt, 
                           f[None, :], 
                           method='PCGPexp')
            
            theta_mle = find_mle(emu, 
                                 cls_data.x, 
                                 x_emu, 
                                 cls_data.real_data, 
                                 cls_data.obsvar, 
                                 1, 
                                 1, 
                                 cls_data.thetalimits, 
                                 is_bias=False)
        
            xnew = ceivarfig(1, 
                      cls_data.x, 
                      cls_data.x,
                      emu, 
                      xt, 
                      f[None, :], 
                      cls_data.real_data, 
                      cls_data.obsvar, 
                      cls_data.thetalimits, 
                      prior_xt,
                      prior_t,
                      thetatest=None, 
                      x_mesh=xmesh,
                      thetamesh=thetamesh, 
                      posttest=ptest,
                      type_init=None,
                      synth_info=cls_data,
                      theta_mle=theta_mle,
                      axis=axs)
            
            xt = np.concatenate((xt, xnew), axis=0)
            f = cls_data.function(xt[:, 0], xt[:, 1])
    else:
        # Acquire new points
        for i in range(nmax-ninit):
            emu = emulator(x_emu, 
                           xt, 
                           f[None, :], 
                           method='PCGPexp')
            
            theta_mle = find_mle(emu,
                                 cls_data.x, 
                                 x_emu, 
                                 cls_data.real_data, 
                                 cls_data.obsvar, 
                                 1, 
                                 1, 
                                 cls_data.thetalimits, 
                                 is_bias=False)

            xnew = ceivarxfig(1, 
                      cls_data.x, 
                      cls_data.x,
                      emu, 
                      xt, 
                      f[None, :], 
                      cls_data.real_data, 
                      cls_data.obsvar, 
                      cls_data.thetalimits, 
                      prior_xt,
                      prior_t,
                      thetatest=None, 
                      x_mesh=xmesh,
                      thetamesh=thetamesh, 
                      posttest=ptest,
                      type_init=None,
                      synth_info=cls_data,
                      theta_mle=theta_mle,
                      axis=axs)
            
            xt = np.concatenate((xt, xnew), axis=0)
            f = cls_data.function(xt[:, 0], xt[:, 1])

plt.savefig('Figure2.jpg', format='jpeg', bbox_inches="tight", dpi=1000)
plt.show()    