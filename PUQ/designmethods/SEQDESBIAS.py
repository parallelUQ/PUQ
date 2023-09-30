import numpy as np
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import multiple_pdfs
from PUQ.designmethods.gen_funcs.CEIVAR import ceivarbias
from PUQ.designmethods.gen_funcs.CEIVARX import ceivarxbias
from PUQ.designmethods.SEQCALsupport import load_H, update_arrays, create_arrays, pad_arrays, select_condition, rebuild_condition
from PUQ.designmethods.utils import collect_data, fit_emulator1d, find_mle, bias_predict
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

def fit(fitinfo, data_cls, args):

    mini_batch = args['mini_batch']
    n_init_thetas = args['n_init_thetas']
    nworkers = args['nworkers']
    AL_type = args['AL']
    seed_n0 = args['seed_n0']
    prior = args['prior']
    max_evals = args['max_evals']
    test_data = args['data_test']
    unknowncov = args['unknowncov']
    theta_torun = args['theta_torun']
    
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
        ('thetamle', float, (1,)),
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
            'unknowncov': unknowncov,
            'theta_torun': theta_torun,
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
    
    for key in persis_info.keys():
        #print(type(persis_info[key]))
        if isinstance(persis_info[key], dict):
            if 'thetamle' in persis_info[key].keys():
                fitinfo['thetamle'] = persis_info[key]['thetamle']

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
        theta_torun     = gen_specs['user']['theta_torun']
        unknowncov      = gen_specs['user']['unknowncov']
        
        # Prior functions
        prior_func_all  = gen_specs['user']['prior']
        prior_func, prior_func_x, prior_func_t = prior_func_all['prior'], prior_func_all['priorx'], prior_func_all['priort']
        
        # Simulation info
        synth_info      = gen_specs['user']['synth_cls']
        obsvar, data, theta_limits, dim, x = synth_info.obsvar, synth_info.real_data, synth_info.thetalimits, synth_info.d, synth_info.x

        # Test data
        test_data = gen_specs['user']['test_data']
        thetatest, posttest, ftest, priortest = None, None, None, None
        if test_data is not None:
            thetatest, th_mesh, x_mesh, ptest, ftest, priortest, ytest = test_data['theta'], test_data['th'], test_data['xmesh'], test_data['p'], test_data['f'], test_data['p_prior'], test_data['y']

        # Additional set
        true_fevals = np.reshape(data[0, :], (1, data.shape[1]))
        x_emu = np.arange(0, 1)[:, None ]
        dx, dt, nmesh = x.shape[1], th_mesh.shape[1], len(x_mesh)
        obs_offset, theta_offset, generated_no = 0, 0, 0
        TV, HD, tag, theta = 1000, 1000, 0, 0
        fevals, pending, prev_pending, complete, prev_complete = None, None, None, None, None
        first_iter, update_model = True, False
        list_id, mlelist = [], []
        
        if AL == None:
            pass
        else:
            acquisition_f = eval(AL)
        
        
        while tag not in [STOP_TAG, PERSIS_STOP]:
            if not first_iter:
                # Update fevals from calc_in
                update_arrays(dim,
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
                emu = fit_emulator1d(x_emu, theta, fevals)
                theta_mle = find_mle(emu, x, x_emu, true_fevals, obsvar, dx, dt, theta_limits, True)
     
                #if (len(theta) % 10 == 0):
                print('mle:', theta_mle)
  
                # Bias prediction 
                bias_pred = bias_predict(emu, theta_mle, x_emu, x, true_fevals, unknowncov)
                
                # Data collect   
                TV, HD = collect_data(emu, bias_pred, x_emu, theta_mle, dt, x_mesh, thetatest, nmesh, ytest, ptest, x, true_fevals, obsvar, synth_info)
   
                # # #
                prev_pending   = pending.copy()
                update_model   = False

            if first_iter:
                n_init = max(n_workers-1, n0)
                theta  = prior_func.rnd(n_init, seed) 
                fevals, pending, prev_pending, complete, prev_complete = create_arrays(dim, n_init)
                            
                H_o    = np.zeros(len(theta), dtype=gen_specs['out'])
                H_o    = load_H(H_o, theta, TV, HD, generated_no, set_priorities=True)
                tag, Work, calc_in = ps.send_recv(H_o)       
                first_iter = False
                generated_no += n_init
                
            else: 
                if select_condition(complete, prev_complete, n_theta=mini_batch, n_initial=n0):

                    prev_complete = complete.copy()
     
                    if AL == None:
                        new_theta = theta_torun[(generated_no-n_init):(generated_no-n_init+mini_batch), :]
                    else:
                        new_theta = acquisition_f(mini_batch, 
                                                  x,
                                                  None,
                                                  emu, 
                                                  theta, 
                                                  fevals, 
                                                  true_fevals, 
                                                  obsvar, 
                                                  theta_limits, 
                                                  prior_func,
                                                  prior_func_t,
                                                  thetatest,
                                                  x_mesh,
                                                  th_mesh,
                                                  priortest,
                                                  bias_pred,
                                                  synth_info,
                                                  theta_mle,
                                                  unknowncov)

                    theta, fevals, pending, prev_pending, complete, prev_complete = \
                        pad_arrays(dim, new_theta, theta, fevals, pending, prev_pending, complete, prev_complete)
        
         
                    H_o = np.zeros(len(new_theta), dtype=gen_specs['out'])
                    H_o = load_H(H_o, new_theta, TV, HD, generated_no, set_priorities=True)
                    tag, Work, calc_in = ps.send_recv(H_o) 
                    generated_no += mini_batch
                    
        
        
        persis_info['thetamle'] =  mlelist
        return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


