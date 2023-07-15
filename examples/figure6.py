import pandas as pd
import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments
from PUQ.prior import prior_dist
from plots import plotline
from test_funcs import unimodal, banana, bimodal, unidentifiable, create_test_data
import time

if __name__ == "__main__":
    
    design_start = time.time()
    args         = parse_arguments()
    print('Running function: ' + args.funcname)
    
    # Choose the test function
    if args.funcname == 'unimodal':
        cls_func = unimodal()
    elif args.funcname == 'banana':
            cls_func = banana()
    elif args.funcname == 'bimodal':
            cls_func = bimodal()
    elif args.funcname == 'unidentifiable':
            cls_func = unidentifiable()
            
    # Create a mesh for test set 
    xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], 50)
    ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], 50)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    th = np.vstack([Xpl.ravel(), Ypl.ravel()])
    setattr(cls_func, 'theta', th.T)
    al_test = designer(data_cls=cls_func, 
                       method='SEQUNIFORM', 
                       args={'mini_batch': 4, 
                             'n_init_thetas': 10,
                             'nworkers': 5,
                             'max_evals': th.shape[1]})
    
    test_data = create_test_data(al_test, cls_func)

    # Set a uniform prior
    prior_func      = prior_dist(dist='uniform')(a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1])

    # Define acquisition functions
    acq_funcs = ['eivar', 'rnd', 'maxvar', 'maxexp']
    datalist = []
    rep_no = 50
    n0 = 10
    # Run over 50 replications
    for seed_id in range(1, rep_no+1):
        for func in acq_funcs:
            print('Running ' + func + ' with seed '  + str(seed_id))
            al_unimodal = designer(data_cls=cls_func, 
                                   method='SEQCAL', 
                                   args={'mini_batch': 1, 
                                         'n_init_thetas': n0,
                                         'nworkers': 2,
                                         'AL': func,
                                         'seed_n0': seed_id, 
                                         'prior': prior_func,
                                         'data_test': test_data, 
                                         'max_evals': 210,
                                         'type_init': None})
            
            TV = al_unimodal._info['TV']
            
            for ms_id, ms in enumerate(TV):
                if ms_id >= n0:
           
                    d = {'TV': ms, 
                         'TV_id': ms_id, 
                         'rep': seed_id, 
                         'batch': 1, 
                         'methods': func, 
                         'worker': 2, 
                         'synth': args.funcname, 
                         'theta_id': ms_id-n0} 
    
                    datalist.append(d)
    
            datalist.append(d)
    df = pd.DataFrame(datalist)  
    
    design_end = time.time()
    print('Elapsed time: ' + str(round(design_end - design_start, 2)))
    
    # Create the plot and save it
    plotline(df, acq_funcs, rep_no, w=2, b=1, s=args.funcname, ylim=[0.000005, 0.00001], idstart=0)
    
    