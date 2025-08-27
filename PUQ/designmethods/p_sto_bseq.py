import numpy as np
from PUQ.designmethods.gen_funcs.batch_allocate_reps import allocate
from PUQ.designmethods.gen_funcs.acquire_new import acquire
from PUQ.designmethods.gen_funcs.batch_acquisition_funcs_support import (
    multiple_pdfs,
    build_emulator,
)
from PUQ.designmethods.batch_support import load_H, update_arrays, create_arrays, pad_arrays, rebuild_condition
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams
import copy
import time

def fit(fitinfo, 
        data_cls,         
        acquisition,
        theta=None,
        prior=None,
        batch_size=None,
        data_test=None,
        nworkers=None,
        max_iter=None,
        trace=0,
        des_init={},
        des_add={},
        alloc_settings={},
        pc_settings={'standardize': True, 'latent': True},
        des_settings={},
        **kwargs):

    out   = data_cls.out
    sim   = data_cls.sim
    
    sim_specs = {
        'sim_f': sim,
        'in': ['thetas'],
        'out': out,
        'user': {
                 'sim_f': data_cls.sim_f
                },
    }

    gen_out = [
        ('thetas', float, data_cls.p),
        ('priority', int),
        ('obs', float, (1,)),
        ('obsvar', float, (1,)),
        ('TV', float),
        ('HD', float),        
        ('AE', float),
        ('time', float),
    ]

    gen_specs = {
        'gen_f': gen_f,
        'persis_in': [o[0] for o in gen_out] + ['f', 'sim_id'],
        'out': gen_out,
        "user": {
            "batch_size": batch_size,  # No. of thetas to generate per step
            "nworkers": nworkers,
            "theta_init": des_init.get('theta'),
            "f_init": des_init.get('f'),
            "theta_add": des_add.get('theta'),
            "synth_cls": data_cls,
            "test_data": data_test,
            "prior": prior,
            "alloc_settings": alloc_settings,
            "acquisition": acquisition,
            "pc_settings": pc_settings,
            "des_settings": des_settings,
            "trace": trace,
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
    for perid, per in enumerate(persis_info):
        persis_info[perid]['rand_stream'] = np.random.default_rng(des_init.get('seed')*(nworkers + 1) + perid)
        #persis_info[perid]['rand_stream'] = np.random.default_rng(perid)

    # Currently just allow gen to exit if mse goes below threshold value
    exit_criteria = {'sim_max': max_iter}  # Now just a set number of sims.

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs=alloc_specs, libE_specs=libE_specs
    )
    
    if data_cls.d == 1:
        fitinfo['f'] = H['f'][:, None]
    else:
        fitinfo['f'] = H['f']
        
    if data_cls.p == 1:
        fitinfo['theta'] = H['thetas']#[:, None]        
    else:
        fitinfo['theta'] = H['thetas']

    f_c     = np.concatenate((des_init.get('f'), fitinfo['f'].T), axis=1)
    print(des_init.get('theta').shape)
    print(fitinfo['theta'].shape)
    t_c     = np.concatenate((des_init.get('theta'), fitinfo['theta']), axis=0)
    
    emu, TV = newiteration(t_c, f_c, data_cls.x, pc_settings, data_test, data_cls.obsvar, data_cls.real_data)
    
    fitinfo['theta']  = t_c
    fitinfo['f']      = f_c
    fitinfo['TViter'] = np.concatenate((H['TV'], np.repeat(TV, batch_size)))
    fitinfo['theta0'] = emu._info['emulist'][0]['X0']
    fitinfo['reps0']  = emu._info['emulist'][0]['mult']

    for key in persis_info.keys():
        if isinstance(persis_info[key], dict):
            if 'additional' in persis_info[key].keys():
                fitinfo['TV'] = persis_info[key]['additional']['TVs']
                fitinfo['TV'].append(TV)
                fitinfo['iter_explore'] = persis_info[key]['additional']['iter_explore']
                fitinfo['iter_exploit'] = persis_info[key]['additional']['iter_exploit']
                fitinfo['time'] = persis_info[key]['additional']['times']         
    return

def newiteration(theta, fevals, x, pc_settings, test_data, obsvar, obs):

    print("hay")
    thetatest, ptest, ftest, priortest = None, None, None, None
    if test_data is not None:
        thetatest, ptest, ftest, priortest = (
            test_data["theta"],
            test_data["p"],
            test_data["f"],
            test_data["p_prior"],
        )
        
    emu = build_emulator(x, theta, fevals, pc_settings) 
    
    print("kkk")
    d = len(x)
    obsvar3d = obsvar.reshape(1, d, d) 
    
    # ntest x d
    pred = emu.predict(x=x, theta=thetatest)
    mu = pred.mean().T 
    S = pred._info['S'] 
    St = np.transpose(S, (2, 0, 1))
    N = St + obsvar3d
    phat = multiple_pdfs(obs, mu, N)
    
    print("hay")

    # Obtain the accuracy on the test set
    if ptest is not None:
        TV = np.mean(np.abs(ptest - phat))
    
    return emu, TV

def onebatch(b, 
             b_new, 
             r_new, 
             emu, 
             prior_func, 
             acqfunc, 
             data_cls, 
             test_data, 
             alloc_settings, 
             pc_settings, 
             des_settings,
             iter_exploit, 
             iter_explore,
             is_exploit,
             is_explore,
             rand_stream,
             trace):

    ivar_exploit, ivar_explore = np.inf, np.inf
    
    emu_original_info = copy.deepcopy(emu._info)
    
    if is_exploit:
        # Allocate existing ones
        allocate_obj = allocate(budget=b,
                                emu_info=emu,
                                prior=prior_func,
                                func_cls=data_cls, 
                                theta_mesh=test_data["theta"], 
                                method=alloc_settings.get('method'),
                                alloc_settings=alloc_settings,
                                rand_stream=rand_stream,
                                trace=trace)
        allocate_obj.allocatereps()
        if not is_explore:
            ivar_exploit = 0
        else:
            ivar_exploit = allocate_obj.ivar_exploit(emu, pc_settings)
    
        r_exploit = allocate_obj.reps
        theta_exploit = allocate_obj.theta
        
    emu._info = emu_original_info       

    if is_explore:
        # Find new ones
        acquire_obj = acquire(bnew=b_new, 
                              rep=r_new,
                              emu=emu, 
                              func_cls=data_cls, 
                              theta_mesh=test_data["theta"], 
                              prior=prior_func,
                              method=acqfunc,
                              nL=des_settings.get('nL'),
                              pc_settings=pc_settings,
                              rand_stream=rand_stream,
                              impute_str=des_settings.get('impute_str'),
                              skip=not is_exploit)
        
        acquire_obj.acquire_new()
    
        ivar_explore = acquire_obj.ivar
        theta_explore = acquire_obj.tnew

    if ivar_exploit <= ivar_explore:
        new_theta = np.repeat(theta_exploit, r_exploit, axis=0)
        iter_exploit += 1
    else:
        new_theta = np.repeat(theta_explore, r_new, axis=0)
        iter_explore += 1

    return new_theta, iter_exploit, iter_explore
    
def gen_f(H, persis_info, gen_specs, libE_info):

        """Generator to select and obviate parameters for calibration."""
        ps              = PersistentSupport(libE_info, EVAL_GEN_TAG)
        rand_stream     = persis_info['rand_stream']

        b               = gen_specs["user"]["batch_size"]
        n_workers       = gen_specs["user"]["nworkers"]

        data_cls        = gen_specs["user"]["synth_cls"]
        test_data       = gen_specs["user"]["test_data"]
        prior_func      = gen_specs["user"]["prior"]
        theta_init      = gen_specs["user"]["theta_init"]
        f_init          = gen_specs["user"]["f_init"]
        theta_add       = gen_specs["user"]["theta_add"]

        alloc_settings  = gen_specs["user"]["alloc_settings"]
        acqfunc         = gen_specs["user"]["acquisition"]
        pc_settings     = gen_specs["user"]["pc_settings"]
        des_settings    = gen_specs["user"]["des_settings"]
        trace           = gen_specs["user"]["trace"]
        
        # data params
        d = data_cls.d
        x = data_cls.x
        y = data_cls.real_data
        Sigma = data_cls.obsvar
        
        # explore params
        rho = alloc_settings.get('rho')
        if rho is not None:
            b_new = int(b*rho)
            r_new = int(b/b_new)

        obs_offset, theta_offset = 0, 0

        fevals, pending, prev_pending, complete, prev_complete = None, None, None, None, None
        first_iter, update_model = True, False
        tag = 0
        iter_explore, iter_exploit = 0, 0
        is_explore, is_exploit = des_settings.get('is_explore'), des_settings.get('is_exploit')
        counter = 0
        TVs, times = [], []
        
        print(f_init.shape)

        while tag not in [STOP_TAG, PERSIS_STOP]:
            
            if not first_iter:
                # Update fevals from calc_in
                update_arrays(d,
                              fevals, 
                              pending, 
                              complete, 
                              calc_in,
                              obs_offset, 
                              theta_offset)
                update_model = rebuild_condition(complete, prev_complete, n_theta=b)
                
                if not update_model:
                    tag, Work, calc_in = ps.recv()
                    if tag in [STOP_TAG, PERSIS_STOP]:
                        break

            if update_model:
                
                if trace == 1:
                    print('Percentage Pending: %0.2f ( %d / %d)' % (100*np.round(np.mean(pending), 4),
                                                                        np.sum(pending),
                                                                        np.prod(pending.shape)))
                    print('Percentage Complete: %0.2f ( %d / %d)' % (100*np.round(np.mean(complete), 4),
                                                                          np.sum(complete),
                                                                          np.prod(pending.shape)))
       
                f_c = np.concatenate((f_init, fevals), axis=1)
                t_c = np.concatenate((theta_init, theta), axis=0)
                
                print(f_c.shape)
                start_emu = time.time()
                emu, TV = newiteration(t_c, f_c, x, pc_settings, test_data, Sigma, y)
                end_emu = time.time()
                TVs.append(TV)

                prev_pending   = pending.copy()
                update_model   = False

            if first_iter:
                
                                
                print("hey")
                
                start_emu = time.time()
                emu, TV = newiteration(theta_init, f_init, x, pc_settings, test_data, Sigma, y)
                end_emu = time.time()
                TVs.append(TV)
                
                print("hey")

                n_init = n_workers - 1
                if theta_add is None:
                    start_batch = time.time()
                    theta, iter_exploit, iter_explore = onebatch(b, 
                                                                 b_new, 
                                                                 r_new, 
                                                                 emu, 
                                                                 prior_func, 
                                                                 acqfunc, 
                                                                 data_cls, 
                                                                 test_data, 
                                                                 alloc_settings, 
                                                                 pc_settings, 
                                                                 des_settings,
                                                                 iter_exploit, 
                                                                 iter_explore,
                                                                 is_exploit,
                                                                 is_explore,
                                                                 rand_stream,
                                                                 trace)
                    end_batch = time.time()
                    times.append(end_batch - start_batch + end_emu - start_emu)
                else:
                    cand_theta = theta_add[counter*b_new:(counter + 1)*b_new, :]
                    theta = np.repeat(cand_theta, r_new, axis=0)
                    
                fevals, pending, prev_pending, complete, prev_complete = create_arrays(d, n_init)
                H_o = np.zeros(len(theta), dtype=gen_specs['out'])
                H_o = load_H(H_o, theta, TV)
                tag, Work, calc_in = ps.send_recv(H_o)   
                first_iter = False
                counter += 1
                
            else: 
                if rebuild_condition(complete, prev_complete, n_theta=b):

                    prev_complete = complete.copy()
                    if theta_add is None:
                        start_batch = time.time()
                        new_theta, iter_exploit, iter_explore = onebatch(b, 
                                                                         b_new, 
                                                                         r_new, 
                                                                         emu, 
                                                                         prior_func, 
                                                                         acqfunc, 
                                                                         data_cls, 
                                                                         test_data, 
                                                                         alloc_settings, 
                                                                         pc_settings, 
                                                                         des_settings,
                                                                         iter_exploit, 
                                                                         iter_explore,
                                                                         is_exploit,
                                                                         is_explore,
                                                                         rand_stream,
                                                                         trace)
                        end_batch = time.time()
                        times.append(end_batch - start_batch + end_emu - start_emu)
                    else:
                        cand_theta = theta_add[counter*b_new:(counter + 1)*b_new, :]
                        new_theta = np.repeat(cand_theta, r_new, axis=0)
                    
                    theta, fevals, pending, prev_pending, complete, prev_complete = \
                        pad_arrays(d, new_theta, theta, fevals, pending, prev_pending, complete, prev_complete)
        
                    H_o = np.zeros(len(new_theta), dtype=gen_specs['out'])
                    H_o = load_H(H_o, new_theta, TV)
                    tag, Work, calc_in = ps.send_recv(H_o) 
                    counter += 1

        persis_info['additional'] = {'TVs':TVs, 'iter_exploit':iter_exploit, 'iter_explore':iter_explore, 'times':times}
        return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
