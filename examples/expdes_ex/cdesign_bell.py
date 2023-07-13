import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test
from smt.sampling_methods import LHS
from test_funcs import bellcurve



seeds = 1
for s in range(seeds):
    cls_data = bellcurve()
    cls_data.realdata(s)
    args         = parse_arguments()

    # Observe
    obsdata(cls_data)

    # # # Create a mesh for test set # # # 
    xt_test, ftest, ptest, thetamesh = create_test(cls_data)
          
    test_data = {'theta': xt_test, 
                 'f': ftest,
                 'p': ptest,
                 'th': thetamesh[:, None],    
                 'xmesh': 0,
                 'p_prior': 1} 

    prior_func      = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 
    # # # # # # # # # # # # # # # # # # # # # 
    
    ninit = 10
    al_unimodal = designer(data_cls=cls_data, 
                           method='SEQCOMPDES', 
                           args={'mini_batch': 1, 
                                 'n_init_thetas': ninit,
                                 'nworkers': 2, 
                                 'AL': 'eivar_exp',
                                 'seed_n0': s,
                                 'prior': prior_func,
                                 'data_test': test_data,
                                 'max_evals': 50,
                                 'type_init': None,
                                 'unknown_var': False,
                                 'design': False})
    
    xt_acq = al_unimodal._info['theta']
    f_acq   = al_unimodal._info['f']
    TV_acq = al_unimodal._info['TV']
    
    
    phat_eivar = fitemu(xt_acq, f_acq[:, None], xt_test, thetamesh, cls_data) 
    plot_EIVAR(xt_acq, cls_data, ninit)
    
    print(np.mean(np.abs(phat_eivar - ptest)))
    
    # LHS 
    sampling = LHS(xlimits=cls_data.thetalimits, random_state=s)
    xt_LHS   = sampling(50)
    f_LHS    = np.zeros(len(xt_LHS))
    for t_id, t in enumerate(xt_LHS):
        f_LHS[t_id] = cls_data.function(xt_LHS[t_id, 0], xt_LHS[t_id, 1])
    
    phat_lhs = fitemu(xt_LHS, f_LHS[:, None], xt_test, thetamesh, cls_data) 
    plot_LHS(xt_LHS, cls_data)
    print(np.mean(np.abs(phat_lhs - ptest)))
    
    # Unif
    t_unif = sps.uniform.rvs(0, 1, size=10)
    xvec = np.tile(cls_data.x.flatten(), len(t_unif))
    xt_unif   = np.concatenate((xvec[:, None], np.repeat(t_unif, len(cls_data.x))[:, None]), axis=1)
    f_unif    = np.zeros(len(xt_unif))
    for t_id, t in enumerate(xt_unif):
        f_unif[t_id] = cls_data.function(xt_unif[t_id, 0], xt_unif[t_id, 1])
    
    phat_unif = fitemu(xt_unif, f_unif[:, None], xt_test, thetamesh, cls_data)
    plot_LHS(xt_unif, cls_data)
    print(np.mean(np.abs(phat_unif - ptest)))
