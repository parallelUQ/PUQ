import numpy as np
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import multiple_pdfs
from PUQ.designmethods.SEQCALsupport import fit_emulator, load_H, update_arrays, create_arrays, pad_arrays, select_condition, rebuild_condition, find_mle, find_mle_bias
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from smt.sampling_methods import LHS
from PUQ.posterior import posterior
from PUQ.surrogate import emulator
import scipy.stats as sps
from PUQ.surrogatemethods.PCGPexp import postpred, postpredbias
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def fit(fitinfo, data_cls, args):

    mini_batch = args['mini_batch']
    n_init_thetas = args['n_init_thetas']
    nworkers = args['nworkers']
    seed_n0 = args['seed_n0']
    prior = args['prior']
    max_evals = args['max_evals']
    test_data = args['data_test']
    theta_torun = args['theta_torun']
    bias = args['bias']
    
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
            'seed_n0': seed_n0,
            'synth_cls': data_cls,
            'test_data': test_data,
            'prior': prior,
            'theta_torun': theta_torun,
            'bias': bias,
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

def bias_predict(emu, theta_mle, x_emu, x, true_fevals):
    typebias = 'lm'
    nx = len(x)
    # Bias prediction #
    xp = np.concatenate((x, np.repeat(theta_mle, nx).reshape(nx, len(theta_mle))), axis=1)
    emupred = emu.predict(x=x_emu, theta=xp)
    mu_sim = emupred.mean()
    var_sim = emupred.var()
    bias = (true_fevals - mu_sim).T

    if typebias == 'lm':
        # Fit linear regression model
        model = LinearRegression()
        emubias = model.fit(x, bias)
    else:
        emubias = emulator(x_emu, 
                           x, 
                           bias.T, 
                           method='PCGPexp')
        
    
    class biaspred:
        def __init__(self, typebias, emubias):
            self.type = typebias
            self.model = emubias

        def predict(self, x):
            if self.type == 'lm':
                return self.model.predict(x).T
            else:
                return self.model.predict(x=x_emu, theta=x).mean()
            
    biasobj = biaspred(typebias, emubias)
    return biasobj


def collect_data(emu, emubias, x_emu, theta_mle, dt, xmesh, xtmesh, nmesh, ytest, ptest, x, obs, obsvar, synth_info):
    
    xtrue_test = np.concatenate((xmesh, np.repeat(theta_mle, nmesh).reshape(nmesh, dt)), axis=1)
    predobj = emu.predict(x=x_emu, theta=xtrue_test)
    fmeanhat, fvarhat = predobj.mean(), predobj.var()

    if emubias == None:
        pred_error = np.mean(np.abs(fmeanhat - ytest))
        pmeanhat, pvarhat = postpred(emu._info, x, xtmesh, obs, obsvar)
        post_error = np.mean(np.abs(pmeanhat - ptest))
    else:
        bmeanhat = emubias.predict(xmesh)
        pred_error = np.mean(np.abs(fmeanhat + bmeanhat - ytest))

        bmeanhat = emubias.predict(x)
        pmeanhat, pvarhat = postpredbias(emu._info, x, xtmesh, obs, obsvar, bmeanhat)
        post_error = np.mean(np.abs(pmeanhat - ptest))

    return pred_error, post_error
                
def gen_f(H, persis_info, gen_specs, libE_info):

        """Generator to select and obviate parameters for calibration."""
        ps              = PersistentSupport(libE_info, EVAL_GEN_TAG)
        rand_stream     = persis_info['rand_stream']
        n0              = gen_specs['user']['n_init_thetas']
        mini_batch      = gen_specs['user']['mini_batch']
        n_workers       = gen_specs['user']['nworkers'] 
        seed            = gen_specs['user']['seed_n0']
        synth_info      = gen_specs['user']['synth_cls']
        test_data       = gen_specs['user']['test_data']
        prior_func_all  = gen_specs['user']['prior']
        theta_torun     = gen_specs['user']['theta_torun']
        isbias = gen_specs['user']['bias']
        
        
        obsvar          = synth_info.obsvar
        data            = synth_info.real_data
        theta_limits    = synth_info.thetalimits
        dim             = synth_info.d 
        x               = synth_info.x
        
        prior_func = prior_func_all['prior']
        prior_func_x = prior_func_all['priorx']
        prior_func_t = prior_func_all['priort']
        
        des = synth_info.des

        thetatest, posttest, ftest, priortest = None, None, None, None
        if test_data is not None:
            thetatest, th_mesh, x_mesh, ptest, ftest, priortest, ytest = test_data['theta'], test_data['th'], test_data['xmesh'], test_data['p'], test_data['f'], test_data['p_prior'], test_data['y']


        true_fevals = np.reshape(data[0, :], (1, data.shape[1]))
        x_emu = np.arange(0, 1)[:, None ]
        
        obs_offset, theta_offset, generated_no = 0, 0, 0
        TV, HD = 1000, 1000
        fevals, pending, prev_pending, complete, prev_complete = None, None, None, None, None
        first_iter = True
        tag = 0
        update_model = False

        list_id = []
        theta = 0
        dx = x.shape[1]
        dt = th_mesh.shape[1]
        nmesh = len(x_mesh)
        mlelist = []
        bias_pred = None
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
                
                emu = emulator(x_emu, 
                               theta, 
                               fevals, 
                               method='PCGPexp')

                if isbias:
                    theta_mle = find_mle_bias(emu, x, x_emu, true_fevals, obsvar, dx, dt, theta_limits)
                    bias_pred = bias_predict(emu, theta_mle, x_emu, x, true_fevals)
                else:
                    theta_mle = find_mle(emu, x, x_emu, true_fevals, obsvar, dx, dt, theta_limits)
                
                mlelist.append(theta_mle)
                if (len(theta) % 10 == 0):
                    print('mle:', theta_mle)

                # Data collect   
                TV, HD = collect_data(emu, bias_pred, x_emu, theta_mle, dt, x_mesh, thetatest, nmesh, ytest, ptest, x, true_fevals, obsvar, synth_info)
                                
                prev_pending   = pending.copy()
                update_model   = False

            if first_iter:
                n_init = max(n_workers-1, n0)
                theta  = theta_torun[0:n_init, :]
                fevals, pending, prev_pending, complete, prev_complete = create_arrays(dim, n_init)
                            
                H_o    = np.zeros(len(theta), dtype=gen_specs['out'])
                H_o    = load_H(H_o, theta, TV, HD, generated_no, set_priorities=True)
                tag, Work, calc_in = ps.send_recv(H_o)       
                first_iter = False
                generated_no += n_init
                
            else: 
                if select_condition(complete, prev_complete, n_theta=mini_batch, n_initial=n0):

                    prev_complete = complete.copy()
     
                    new_theta = theta_torun[generated_no:(generated_no+mini_batch), :]

                    theta, fevals, pending, prev_pending, complete, prev_complete = \
                        pad_arrays(dim, new_theta, theta, fevals, pending, prev_pending, complete, prev_complete)
        
                    H_o = np.zeros(len(new_theta), dtype=gen_specs['out'])
                    H_o = load_H(H_o, new_theta, TV, HD, generated_no, set_priorities=True)
                    tag, Work, calc_in = ps.send_recv(H_o) 
                    generated_no += mini_batch
                    
        
        
        persis_info['thetamle'] =  mlelist
        return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

