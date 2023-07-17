import numpy as np
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import multiple_pdfs
from PUQ.designmethods.gen_funcs.acquisition_funcs_des import eivar_exp, eivar_new_exp, eivar_new_exp_mat
from PUQ.designmethods.SEQCALsupport import fit_emulator, load_H, update_arrays, create_arrays, pad_arrays, select_condition, rebuild_condition
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from smt.sampling_methods import LHS
from PUQ.posterior import posterior
from PUQ.surrogate import emulator
import scipy.stats as sps
import scipy.optimize as spo
import matplotlib.pyplot as plt

def fit(fitinfo, data_cls, args):

    mini_batch = args['mini_batch']
    n_init_thetas = args['n_init_thetas']
    nworkers = args['nworkers']
    AL_type = args['AL']
    seed_n0 = args['seed_n0']
    prior = args['prior']
    max_evals = args['max_evals']
    test_data = args['data_test']
    
    
    out = data_cls.out
    sim_f = data_cls.sim
    
    
    sim_specs = {
        'sim_f': sim_f,
        'in': ['thetas'],
        'out': out,
        'user': {
                 'function': data_cls.function
                },
    }

    gen_out = [
        ('thetas', float, data_cls.p),
        ('priority', int),
        ('obs', float, (1,)),
        ('obsvar', float, (1,)),
        ('TV', float),
        ('HD', float),
    ]

    gen_specs = {
        'gen_f': gen_f,
        'persis_in': [o[0] for o in gen_out] + ['f', 'sim_id'],
        'out': gen_out,
        'user': {
            'n_init_thetas': n_init_thetas,  # Num thetas in initial batch
            'mini_batch': mini_batch,  # No. of thetas to generate per step
            'nworkers': nworkers,
            'AL': AL_type,
            'seed_n0': seed_n0,
            'synth_cls': data_cls,
            'test_data': test_data,
            'prior': prior,
            'type_init': args['type_init'],
            'unknown_var': args['unknown_var'],
            'design': args['design']
        },
    }

    alloc_specs = {
        'alloc_f': alloc_f,
        'user': {
            'init_sample_size': 0,
            'async_return': True,  # True = Return results to gen as they come in (after sample)
            'active_recv_gen': True,  # Persistent gen can handle irregular communications
        },
    }
    libE_specs = {'nworkers': nworkers, 'comms': 'local'}
    #libE_specs = {'nworkers': nworkers, 'comms': 'local', 'sim_dirs_make': True,
    #              'sim_dir_copy_files': [os.path.join(os.getcwd(), '48Ca_template.in')]}
                                  
      
    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Currently just allow gen to exit if mse goes below threshold value
    exit_criteria = {'sim_max': max_evals}  # Now just a set number of sims.

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs=alloc_specs, libE_specs=libE_specs
    )
    

    fitinfo['f'] = H['f']
    fitinfo['theta'] = H['thetas']
    fitinfo['TV'] = H['TV']
    fitinfo['HD'] = H['HD']
    return

                
def gen_f(H, persis_info, gen_specs, libE_info):

        """Generator to select and obviate parameters for calibration."""
        ps              = PersistentSupport(libE_info, EVAL_GEN_TAG)
        rand_stream     = persis_info['rand_stream']
        n0              = gen_specs['user']['n_init_thetas']
        mini_batch      = gen_specs['user']['mini_batch']
        n_workers       = gen_specs['user']['nworkers'] 
        AL              = gen_specs['user']['AL']
        seed            = gen_specs['user']['seed_n0']
        synth_info      = gen_specs['user']['synth_cls']
        test_data       = gen_specs['user']['test_data']
        prior_func      = gen_specs['user']['prior']
        type_init       = gen_specs['user']['type_init']
        unknown_var     = gen_specs['user']['unknown_var']
        design          = gen_specs['user']['design']
        
        obsvar          = synth_info.obsvar
        data            = synth_info.real_data
        theta_limits    = synth_info.thetalimits
        
   
        des = synth_info.des
     
        theta_torun     = synth_info.theta_torun

        thetatest, posttest, ftest, priortest = None, None, None, None
        if test_data is not None:
            thetatest, th_mesh, x_mesh, posttest, ftest, priortest = test_data['theta'], test_data['th'], test_data['xmesh'], test_data['p'], test_data['f'], test_data['p_prior']


        true_fevals = np.reshape(data[0, :], (1, data.shape[1]))
        n_x     = synth_info.d 
        x       = synth_info.x
        real_x  = synth_info.real_x
        n_x_des    = len(x)
        x_emu      = np.arange(0, 1)[:, None ]
        
        x_u = 1*x
        true_fevals_u = 1*true_fevals
        obsvar_u = 1*obsvar
        
        obs_offset, theta_offset, generated_no = 0, 0, 0
        TV, HD = 1000, 1000
        fevals, pending, prev_pending, complete, prev_complete = None, None, None, None, None
        first_iter = True
        tag = 0
        update_model = False
        acquisition_f = eval(AL)
        list_id = []
        emubias = None
        theta = 0
        
        while tag not in [STOP_TAG, PERSIS_STOP]:
            if not first_iter:
                # Update fevals from calc_in
                update_arrays(n_x,
                              fevals, 
                              pending, 
                              complete, 
                              calc_in,
                              obs_offset, 
                              theta_offset,
                              list_id)
                update_model = rebuild_condition(complete, prev_complete, n_theta=mini_batch, n_initial=n0)
                
                if not update_model:
                    tag, Work, calc_in = ps.recv()
                    if tag in [STOP_TAG, PERSIS_STOP]:
                        break

            if update_model:

                emu = emulator(x_emu, 
                               theta, 
                               fevals, 
                               method='PCGPexp')

 

                prev_pending   = pending.copy()
                update_model   = False

            if first_iter:
               # print('Selecting theta for the first iteration...\n')

                n_init = max(n_workers-1, n0)

                if type_init == 'LHS':
                    sampling = LHS(xlimits=theta_limits, random_state=seed)
                    theta  = sampling(n_init)
                elif type_init == 'to_run':
                    theta  = theta_torun[0:n_init, :]
                else:
                    theta  = prior_func.rnd(n_init, seed) 
                

                fevals, pending, prev_pending, complete, prev_complete = create_arrays(n_x, n_init)
                            
                H_o    = np.zeros(len(theta), dtype=gen_specs['out'])
                H_o    = load_H(H_o, theta, TV, HD, generated_no, set_priorities=True)
                tag, Work, calc_in = ps.send_recv(H_o)       
                first_iter = False
                generated_no += n_init
                
            else: 
                if select_condition(complete, prev_complete, n_theta=mini_batch, n_initial=n0):

                    prev_complete = complete.copy()
                    
                    if type_init == 'to_run':
                        new_theta = theta_torun[generated_no:(generated_no+mini_batch), :]
                    else:
                        new_theta = acquisition_f(mini_batch, 
                                                  x_u,
                                                  real_x,
                                                  emu, 
                                                  theta, 
                                                  fevals, 
                                                  true_fevals_u, 
                                                  obsvar_u, 
                                                  theta_limits, 
                                                  prior_func,
                                                  thetatest,
                                                  th_mesh,
                                                  priortest,
                                                  type_init)

                    theta, fevals, pending, prev_pending, complete, prev_complete = \
                        pad_arrays(n_x, new_theta, theta, fevals, pending, prev_pending, complete, prev_complete)
        
         
                    H_o = np.zeros(len(new_theta), dtype=gen_specs['out'])
                    H_o = load_H(H_o, new_theta, TV, HD, generated_no, set_priorities=True)
                    tag, Work, calc_in = ps.send_recv(H_o) 
                    generated_no += mini_batch
                    
        
        

        return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

