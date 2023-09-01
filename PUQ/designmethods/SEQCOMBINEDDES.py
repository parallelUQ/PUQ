import numpy as np
from PUQ.designmethods.gen_funcs.acquisition_funcs_des import eivar_des_updated, eivar_des_updated2
from PUQ.designmethods.gen_funcs.PMAXVAR import pmaxvar, pmaxvar_upd, pmaxvar2
from PUQ.designmethods.gen_funcs.PEIVAReff import peivareff, peivarefffig
from PUQ.designmethods.gen_funcs.CEIVAR import ceivar
from PUQ.designmethods.gen_funcs.CEIVARX import ceivarx
from PUQ.designmethods.gen_funcs.PIMSE import pimse
from PUQ.designmethods.SEQCALsupport import fit_emulator, load_H, update_arrays, create_arrays, pad_arrays, select_condition, rebuild_condition, find_mle
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
from PUQ.designmethods.gen_funcs.PEIVARinit import peivarinitial, pmaxvarinitial
from PUQ.surrogatemethods.PCGPexp import  postpred

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
            'prior': prior
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
    
    #print(persis_info)

    fitinfo['f'] = H['f']
    fitinfo['theta'] = H['thetas']
    fitinfo['TV'] = H['TV']
    fitinfo['HD'] = H['HD']

    for key in persis_info.keys():
        #print(type(persis_info[key]))
        if isinstance(persis_info[key], dict):
            if 'des' in persis_info[key].keys():
                fitinfo['des'] = persis_info[key]['des']
            if 'thetamle' in persis_info[key].keys():
                fitinfo['thetamle'] = persis_info[key]['thetamle']
                
    return

def collect_data(emu, x_emu, theta_mle, dt, xmesh, xtmesh, nmesh, ytest, ptest, x, obs, obsvar):
    
    xtrue_test = np.concatenate((xmesh, np.repeat(theta_mle, nmesh).reshape(nmesh, dt)), axis=1)
    
    predobj = emu.predict(x=x_emu, theta=xtrue_test)
    ymeanhat, yvarhat = predobj.mean(), predobj.var()
    
    pred_error = np.mean(np.abs(ymeanhat - ytest))
    
    #pmeanhat, pvarhat = postpred(emu._info, x, xtmesh, obs, obsvar)
    
    #post_error = np.mean(np.abs(pmeanhat - ptest))
    
    return pred_error, None
                
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
        prior_func_all  = gen_specs['user']['prior']

        obsvar          = synth_info.obsvar
        data            = synth_info.real_data
        theta_limits    = synth_info.thetalimits
        
        prior_func = prior_func_all['prior']
        prior_func_x = prior_func_all['priorx']
        prior_func_t = prior_func_all['priort']
        
        real_data_rep   = None

        thetatest, posttest, ftest, priortest = None, None, None, None
        if test_data is not None:
            thetatest, th_mesh, x_mesh, ptest, ftest, priortest, ytest = test_data['theta'], test_data['th'], test_data['xmesh'], test_data['p'], test_data['f'], test_data['p_prior'], test_data['y']

        n_x     = synth_info.d 
        x       = synth_info.x
        x_emu   = np.arange(0, 1)[:, None ]
        dx = synth_info.dx 
        dt = synth_info.true_theta.shape[0]


        if synth_info.nodata:
            true_fevals, x_u, true_fevals_u, obsvar_u, des = None, None, None, None, []
            nodesign = True
        else:
            true_fevals = np.reshape(data[0, :], (1, data.shape[1]))
            x_u = 1*x
            true_fevals_u = 1*true_fevals
            obsvar_u = 1*obsvar
            des = synth_info.des
            nodesign = False

        obs_offset, theta_offset, generated_no = 0, 0, 0
        TV, HD = 1000, 1000
        fevals, pending, prev_pending, complete, prev_complete = None, None, None, None, None
        first_iter = True
        tag = 0
        update_model = False
        acquisition_f = eval(AL)
        list_id = []
        theta = 0
        
        nf_init = 0
        mlelist = []
        
        batchcounter, batchfield = 0, 0
        nmesh = len(x_mesh)
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

                emu = emulator(x_emu, 
                               theta, 
                               fevals, 
                               method='PCGPexp')

                theta_mle = find_mle(emu, x_u, x_emu, true_fevals_u, obsvar_u, dx, dt, theta_limits)
                mlelist.append(theta_mle)
                print('mle:', theta_mle)
                
                TV, HD = collect_data(emu, x_emu, theta_mle, dt, x_mesh, thetatest, nmesh, ytest, ptest, x_u, true_fevals_u, obsvar_u)

                new_field = True if ((theta.shape[0] % 10) == 0) and (theta.shape[0] > n_init) else False
            
                if new_field:
                    batchcounter += 1
                    des = peivareff(prior_func, 
                                    prior_func_x, 
                                    emu, 
                                    x_emu, 
                                    theta_mle, 
                                    x_mesh, 
                                    th_mesh,
                                    synth_info, 
                                    des, 
                                    batchfield=batchfield, 
                                    batchcounter=batchcounter)
                    
                x_u           = np.array([e['x'] for e in des])
                
                #if batchfield == batchcounter:
                #    for e in des:
                #        if e['isreal'] == 'No':
                #            xvar = synth_info.realvar(e['x'])
                #            y_new = synth_info.genobsdata(e['x'], xvar) 
                #            e['feval'] = [y_new]
                #            e['isreal'] = 'Yes'
                #    batchcounter = 0
                        
                true_fevals_u = np.array([np.mean(e['feval']) for e in des])[None, :]
                obsvar_u      = np.diag(synth_info.realvar(x_u))
                prev_pending   = pending.copy()
                update_model   = False
                
                    
            if first_iter:
                n_init = max(n_workers-1, n0)
                theta  = prior_func.rnd(n_init, seed) 
                fevals, pending, prev_pending, complete, prev_complete = create_arrays(n_x, n_init)  
                H_o    = np.zeros(len(theta), dtype=gen_specs['out'])
                H_o    = load_H(H_o, theta, TV, HD, generated_no, set_priorities=True)
                tag, Work, calc_in = ps.send_recv(H_o)       
                first_iter = False
                generated_no += n_workers-1
                
            else: 
                if select_condition(complete, prev_complete, n_theta=mini_batch, n_initial=n0):
                    print('Computer design')
                    prev_complete = complete.copy()
                    new_theta = acquisition_f(mini_batch, 
                                              x_u,
                                              None,
                                              emu, 
                                              theta, 
                                              fevals, 
                                              true_fevals_u, 
                                              obsvar_u, 
                                              theta_limits, 
                                              prior_func,
                                              prior_func_t,
                                              thetatest,
                                              x_mesh,
                                              th_mesh,
                                              priortest,
                                              None,
                                              synth_info,
                                              theta_mle)
                    

                    theta, fevals, pending, prev_pending, complete, prev_complete = \
                        pad_arrays(n_x, new_theta, theta, fevals, pending, prev_pending, complete, prev_complete)
        
         
                    H_o = np.zeros(len(new_theta), dtype=gen_specs['out'])
                    H_o = load_H(H_o, new_theta, TV, HD, generated_no, set_priorities=True)
                    tag, Work, calc_in = ps.send_recv(H_o) 
                    generated_no += mini_batch

        persis_info['des'] =  des
                
        persis_info['thetamle'] =  mlelist
        return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

