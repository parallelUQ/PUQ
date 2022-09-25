import numpy as np
import os
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import get_emuvar, multiple_pdfs
from PUQ.designmethods.gen_funcs.acquisition_funcs import maxvar, eivar, maxexp, hybrid, rnd
from PUQ.designmethods.SEQCALsupport import fit_emulator, load_H, update_arrays, create_arrays, pad_arrays, select_condition, rebuild_condition
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
import matplotlib.pyplot as plt
from PUQ.prior import prior_dist

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
        p_dist          = gen_specs['user']['prior']
        
        obsvar          = synth_info.obsvar
        data            = synth_info.real_data
        theta_limits    = synth_info.thetalimits
        
        prior_func      = prior_dist(dist=p_dist)
        
        thetatest, posttest, ftest = None, None, None
        
        if test_data is not None:
            thetatest, posttest, ftest = test_data['theta'], test_data['p'], test_data['f']
        

        true_fevals = np.reshape(data[0, :], (1, data.shape[1]))

        n_x     = synth_info.d 
        n_realx = true_fevals.shape[1]
        x       = synth_info.x
        real_x  = synth_info.real_x

        obs_offset, theta_offset, generated_no = 0, 0, 0
        TV, HD = 1000, 1000
        fevals, pending, prev_pending, complete, prev_complete = None, None, None, None, None
        first_iter = True
        tag = 0

        obsvar3d   = obsvar.reshape(1, n_x, n_x)
        update_model = False
        acquisition_f = eval(AL)
        list_id = []

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
                print('Updating model...\n')

                print('Percentage Pending: %0.2f ( %d / %d)' % (100*np.round(np.mean(pending), 4),
                                                                np.sum(pending),
                                                                np.prod(pending.shape)))
                print('Percentage Complete: %0.2f ( %d / %d)' % (100*np.round(np.mean(complete), 4),
                                                                 np.sum(complete),
                                                                 np.prod(pending.shape)))
                
                emu            = fit_emulator(x, theta, fevals, theta_limits)
                prev_pending   = pending.copy()
                update_model   = False
                
                if test_data is not None:
                    emupredict     = emu.predict(x=x, theta=thetatest)
                    emumean        = emupredict.mean()
                    emuvar, is_cov = get_emuvar(emupredict)
                    emumeanT       = emumean.T
                    emuvarT        = emuvar.transpose(1, 0, 2)
                    var_obsvar1    = emuvarT + obsvar3d 
    
                    posttesthat    = multiple_pdfs(true_fevals, 
                                                   emumeanT[:, real_x.flatten()], 
                                                   var_obsvar1[:, real_x, real_x.T])
                    
                    TV = np.mean(np.abs(posttest - posttesthat))
                    HD = np.sqrt(0.5*np.mean((np.sqrt(posttesthat) - np.sqrt(posttest))**2))     

            if first_iter:
                print('Selecting theta for the first iteration...\n')

                n_init = max(n_workers-1, n0)
                theta  = prior_func(n_init, theta_limits, seed)
                fevals, pending, prev_pending, complete, prev_complete = create_arrays(n_x, n_init)
                            
                H_o    = np.zeros(len(theta), dtype=gen_specs['out'])
                H_o    = load_H(H_o, theta, TV, HD, generated_no, set_priorities=True)
                tag, Work, calc_in = ps.send_recv(H_o)       
                first_iter = False
                generated_no += n_workers-1
                
            else: 
                if select_condition(complete, prev_complete, n_theta=mini_batch, n_initial=n0):
                    print('Selecting theta...\n')
                
                    prev_complete = complete.copy()
                    new_theta = acquisition_f(mini_batch, 
                                              x,
                                              real_x,
                                              emu, 
                                              theta, 
                                              fevals, 
                                              true_fevals, 
                                              obsvar, 
                                              theta_limits, 
                                              prior_func,
                                              thetatest,
                                              posttest)

                    theta, fevals, pending, prev_pending, complete, prev_complete = \
                        pad_arrays(n_x, new_theta, theta, fevals, pending, prev_pending, complete, prev_complete)
        
         
                    H_o = np.zeros(len(new_theta), dtype=gen_specs['out'])
                    H_o = load_H(H_o, new_theta, TV, HD, generated_no, set_priorities=True)
                    tag, Work, calc_in = ps.send_recv(H_o) 
                    generated_no += mini_batch

        return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

