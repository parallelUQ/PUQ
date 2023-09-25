import pickle
import numpy as np
import datetime as dt
from interventions import create_intLevel, form_interventions
from SEIYAHRD_sim_filter import simulate_p
from InterventionsMIP import config, logger, output_path
from utils import profile_log, print_profiling_log
from threshold_policy import policy_multi_iterator_filter, stoch_simulation_iterator_filter

datetime_formater = '%Y-%m-%d %H:%M:%S'
date_formater = '%Y-%m-%d'

def policy_search_filter(instance,
                  tiers,
                  obj_func,
                  n_replicas_train=100,
                  n_replicas_test=100,
                  instance_name=None,
                  policy_class='constant',
                  policy=None,
                  mp_pool=None,
                  crn_seeds=[],
                  unique_seeds_ori=[],
                  forcedOut_tiers=None,
                  redLimit=1000,
                  after_tiers=[0,1,2,3,4],
                  policy_field="IYIH",
                  policy_ub=None):
    '''
        TODO: Write proper docs
    '''
    
    #    Set up for policy search: build interventions according to input tiers
    fixed_TR = list(filter(None, instance.cal.fixed_transmission_reduction))
    tier_TR = [item['transmission_reduction'] for item in tiers]
    uniquePS = sorted(np.unique(np.append(fixed_TR, np.unique(tier_TR))))
    sc_levels = np.unique([tier['school_closure'] for tier in tiers] + [0, 1])
    fixed_CO = list(filter(None, instance.cal.fixed_cocooning))
    co_levels = np.unique(np.append([tier['cocooning'] for tier in tiers], np.unique(fixed_CO)) + [0])
    intervention_levels = create_intLevel(sc_levels, co_levels, uniquePS)
    interventions_train = form_interventions(intervention_levels, instance.epi, instance.N)
    
    # Build an iterator of all the candidates to be simulated by simulate_p
    sim_configs = policy_multi_iterator_filter(instance,
                                               tiers,
                                               obj_func,
                                               interventions_train,
                                               policy_class=policy_class,
                                               fixed_policy=policy,
                                               fo_tiers=forcedOut_tiers,
                                               redLimit=redLimit,
                                               after_tiers=after_tiers,
                                               policy_field=policy_field,
                                               policy_ub=policy_ub)
    # Launch parallel simulation
    all_outputs = simulate_p(mp_pool, sim_configs)
    logger.info(f'Simulated candidates: {len(all_outputs)}: {profile_log["simulate_p"]}')
    
    # Search of the best feasible candidate
    best_cost, best_sim, best_policy, best_params = np.Inf, None, None, None
    hosp_benchmark = None
    if len(all_outputs) == 1:
        # Skip search if there is only one candidate
        sim_output, cost, best_policy, seed_0, kwargs_out = all_outputs[0]
        best_cost = cost
        best_sim = sim_output
        best_params = kwargs_out
        cost_record = {}
        best_id = 0
        if config['hosp_aligned_real']:
            if hosp_benchmark is None:
                hosp_benchmark = instance.cal.real_hosp
        else:
            if hosp_benchmark is None:
                hosp_benchmark = [sim_output['IHT'][t].sum() for t in range(len(instance.cal.real_hosp))]
    else:
        # Check feasibility and optimality of all the simulated candidates
        #   - Feasibility: chance constraint feasible
        #   - Optimality: according to obj_func, but ignoring penalties
        
        SRS_pruned = 0  # Tally of candidates pruned by square root staffing
        found_feasible_policy = False  # Feasible solution found flag
        chance_constrain_vio = np.inf  # Min violation of the CC
        best_cost_inf = np.inf  # Cost of the least inf solution
        cost_record = {}  # Costs information (lockdown cost, over-capacity cost) for each candidate
        for ix, output_i in enumerate(all_outputs):
            # Loop through all the
            sim_output, cost, policy_i, seed_i, kwargs_out = output_i
            cost_record_ij = []
            if config['hosp_aligned_real']:
                if hosp_benchmark is None:
                    hosp_benchmark = instance.cal.real_hosp
            else:
                if hosp_benchmark is None:
                    hosp_benchmark = [sim_output['IHT'][t].sum() for t in range(len(instance.cal.real_hosp))]
            if cost < np.inf or not found_feasible_policy:
                # Staffing rule feasible
                logger.info(f'Considering: {str(policy_i)}')
                kwargs_out['opt_phase'] = False  # not longer optimizing
                kwargs_out['infeasible_penalty'] = False  # don't penalize with staffing rule
                kwargs_out['over_capacity_cost'] = config['over_capacity_cost']  # add the over capacity penalty
                kwargs_out['obj_over_included'] = config['obj_over_included']  # whether the objective function includes over-capacity cost
                kwargs_out['sim_method'] = config['sim_method']
                kwargs_out['fo_tiers'] = forcedOut_tiers
                kwargs_out['changed_tiers'] = False
                kwargs_out['redLimit'] = redLimit
                kwargs_out['after_tiers'] = after_tiers
                kwargs_out['active_intervention'] = config['active_intervention']
                kwargs_out['extra_capacity_cost'] = config['extra_capacity_cost']
                kwargs_out['capacity_setup_cost'] = config['capacity_setup_cost']
                kwargs_out['icu_trigger'] = config['icu_trigger']
                kwargs_out['icu_capacity_cost'] = config['icu_capacity_cost']
                
                try:
                    start_date = dt.datetime.strptime(config["rd_start"], datetime_formater)
                    kwargs_out["rd_start"] = instance.cal.calendar_ix[start_date]
                    end_date = dt.datetime.strptime(config["rd_end"], datetime_formater)
                    kwargs_out["rd_end"] = instance.cal.calendar_ix[end_date]
                    kwargs_out["rd_rate"] = config["rd_rate"]
                except:
                    kwargs_out["rd_start"] = -1
                    kwargs_out["rd_end"] = -1
                    kwargs_out["rd_rate"] = 1
                
                stoch_outputs_i = []  # List of valid samples if policy_i
                crn_seeds_i = []  # Common random number seed of policy_i (set once)
                deviation_output = []  # print out the deviation of the seeds
                total_train_reps = 0  # Total samples executed in the filtering procedure
                
                real_hosp = hosp_benchmark
                # =================================================
                # Sample until required number of replicas achieved
                while len(stoch_outputs_i) < n_replicas_train:
                    chunksize = 4 if mp_pool is None else mp_pool._processes
                    chunksize = chunksize if crn_seeds == [] else n_replicas_train
                    total_train_reps += chunksize
                    if crn_seeds == []:
                        # no input seeds
                        seed_shift_var = total_train_reps
                        crn_input = None
                        chunkinput = chunksize
                    else:
                        seed_shift_var = 0
                        crn_input = crn_seeds[total_train_reps-chunksize:total_train_reps]
                        if len(crn_input) == 0:
                            # if the given seeds are run out, need to generate new seeds
                            crn_input = None
                            chunkinput = chunksize
                            seed_shift_var = crn_seeds[-1] + 1 + total_train_reps
                        else:
                            chunkinput = len(crn_input)
                    
                    rsq_thr = config["rsq_threshold"]
 
                    change_dates = [dt.datetime(2020, 3, 23, 0, 0), dt.datetime(2020, 5, 20, 0, 0), dt.datetime(2020, 6, 25, 0, 0), dt.datetime(2020, 8, 19, 0, 0), dt.datetime(2020, 10, 7, 0, 0)]
                    stoch_next = []
                    seed_next = []
                    for k in range(0, len(change_dates)):  
                        if k == 0:
                            kwargs_out['particle_filtering'] = True
                            kwargs_out['end_date'] = change_dates[0]
                            kwargs_out['start_date'] = instance.start_date
                            seed_next = crn_input
                            shift = seed_shift_var
                            replica_no = chunksize
                        else:
                            kwargs_out['end_date'] = change_dates[k]
                            kwargs_out['start_date'] = change_dates[k - 1] + dt.timedelta(days=1)
                            shift = 0
                            replica_no = len(seed_next)

                        out_sample_configs = stoch_simulation_iterator_filter(instance,
                                                                              policy_i,
                                                                              obj_func,
                                                                              interventions_train,
                                                                              ex_res=stoch_next,
                                                                              seed_shift=shift,
                                                                              crn_seeds=seed_next,
                                                                              n_replicas=replica_no,
                                                                              det_sample_path=False,
                                                                              **kwargs_out)
                        out_sample_outputs = simulate_p(mp_pool, out_sample_configs)
                        T = 1 + (kwargs_out['end_date'] - instance.start_date).days
                        seed_temp = []
                        r_sq_path = []
                        stoch_outputs_temp = []
                        ids = []
                        for sample_ij in out_sample_outputs:
                            sim_j, cost_j, policy_j, seed_j, kwargs_j = sample_ij
                            stoch_outputs_temp.append(sample_ij)
                            seed_temp.append(seed_j)
                            IH_j = np.sum(sim_j['IH'][0:T] + sim_j['ICU'][0:T], axis = (1,2))
                            r_sq_path.append(1 - sum(np.square(real_hosp[0:T]- IH_j))/sum(np.square(real_hosp[0:T] - np.mean(real_hosp[0:T]))))
            
                        candidate_ids = [i for i,x in enumerate(r_sq_path) if x > rsq_thr]
                        r_sq_path = [r_sq_path[i] for i in candidate_ids]
  
                        if len(r_sq_path) > 0:
                            ids = candidate_ids
                            stoch_next = [stoch_outputs_temp[i] for i in ids] 
                            seed_next = [seed_temp[i] for i in ids] 
                        else:
                            break
                    
                    if len(ids) > 0:
                        kwargs_out['end_date'] = instance.end_date
                        kwargs_out['start_date'] = change_dates[-1] + dt.timedelta(days=1)     
                        out_sample_configs = stoch_simulation_iterator_filter(instance,
                                                                              policy_i,
                                                                              obj_func,
                                                                              interventions_train,
                                                                              ex_res=stoch_next,
                                                                              seed_shift=0,
                                                                              crn_seeds=seed_next,
                                                                              n_replicas=len(ids),
                                                                              det_sample_path=False,
                                                                              **kwargs_out)
                        out_sample_outputs = simulate_p(mp_pool, out_sample_configs)
                        for sample_ij in out_sample_outputs:
                            sim_j, cost_j, policy_j, seed_j, kwargs_j = sample_ij
                            stoch_outputs_i.append(sample_ij)
                            cost_record_ij.append(kwargs_j["cost_info"])
                            crn_seeds_i.append(seed_j)
                            if len(stoch_outputs_i) == n_replicas_train:
                                break
                 
                # Save CRN seeds for all policies yet to be evaluated
                if crn_seeds == []:
                    assert len(np.unique(crn_seeds_i)) == n_replicas_train
                    crn_seeds = crn_seeds_i.copy()
                    logger.info(f'\tCRN SEEDS {str(crn_seeds)}, Deviation {str(deviation_output)}')
                # End of samples filtering procedure
                # =================================================
                
                # Check feasibility with valid traning samples
                logger.info(f'\tgot {len(stoch_outputs_i)} replicas sampling {total_train_reps}')
                stoch_replicas = [rep_i[0] for rep_i in stoch_outputs_i]
                stoch_costs = [rep_i[1] for rep_i in stoch_outputs_i]
                cost_record[str(policy_i)] = cost_record_ij
                if config["icu_trigger"]:
                    infeasible_replicas = np.sum([
                        np.any(stoch_replicas[rep_i]['ICU'].sum(axis=(1, 2)) > instance.icu)
                        for rep_i in range(len(stoch_outputs_i))
                    ])
                    IH_feasible = infeasible_replicas <= int(config['chance_constraint_epsilon'] * n_replicas_train)
                else:
                    infeasible_replicas = np.sum([
                        np.any(stoch_replicas[rep_i]['IH'].sum(axis=(1, 2)) > instance.hosp_beds)
                        for rep_i in range(len(stoch_outputs_i))
                    ])
                    IH_feasible = infeasible_replicas <= int(config['chance_constraint_epsilon'] * n_replicas_train)
                expected_cost = np.mean(stoch_costs)
                logger.info(
                    f'\tInf reps: {infeasible_replicas}  Expected Cost: {expected_cost:.0f} best cost: {best_cost}')
                
                # Update incunbent solution: best_policy
                if not found_feasible_policy and not IH_feasible:
                    # Accept infeasible policies if no feasible policy found yet
                    # If an infeasible policy is accepted, it is the least infeasible
                    cc_vio = infeasible_replicas - int(config['chance_constraint_epsilon'] * n_replicas_train)
                    if cc_vio < chance_constrain_vio or (cc_vio == chance_constrain_vio and expected_cost < best_cost_inf):
                        chance_constrain_vio = cc_vio
                        best_sim, best_cost_inf, best_policy, best_params = sim_output, expected_cost, policy_i, kwargs_out
                        best_id = ix
                        print('Least inf : ', policy_i, '  ', cc_vio, '   ', expected_cost)
                else:
                    # Feasible solution replace incumbent according to expected cost
                    if expected_cost < best_cost and IH_feasible:
                        best_sim = sim_output
                        best_cost = expected_cost
                        best_policy = policy_i
                        best_params = kwargs_out
                        best_id = ix
                        found_feasible_policy = True  # Flag update, inf policies not longer accepted
                        logger.info(
                            f'\tNew feasible solution -> inf reps {infeasible_replicas} : exp. cost: {expected_cost}')
            else:
                # Policy infeasible w.r.t square root staffing rule, solution discarted.
                logger.info(f'Discarted: {str(policy_i)}')
                SRS_pruned += 1
        logger.info(f'SRS pruned {SRS_pruned} out of {len(all_outputs)} candidates')
    print_profiling_log(logger)
    
    # ===================================================================
    # Final evaluation of the best policy using sample filter for samples
    best_params['opt_phase'] = False  # not longer optimizing
    best_params['infeasible_penalty'] = False  # don't penalize with staffing rule
    best_params['over_capacity_cost'] = config['over_capacity_cost']  # add the over capacity penalty
    best_params['obj_over_included'] = config['obj_over_included']
    best_params['sim_method'] = config['sim_method']
    best_params['active_intervention'] = config['active_intervention']
    best_params['fo_tiers'] = forcedOut_tiers
    best_params['changed_tiers'] = False
    best_params['redLimit'] = redLimit
    best_params['after_tiers'] = after_tiers
    best_params['extra_capacity_cost'] = config['extra_capacity_cost']
    best_params['capacity_setup_cost'] = config['capacity_setup_cost']
    best_params['icu_trigger'] = config['icu_trigger']
    best_params['icu_capacity_cost'] = config['icu_capacity_cost']
    try:
        start_date = dt.datetime.strptime(config["rd_start"], datetime_formater)
        best_params["rd_start"] = instance.cal.calendar_ix[start_date]
        end_date = dt.datetime.strptime(config["rd_end"], datetime_formater)
        best_params["rd_end"] = instance.cal.calendar_ix[end_date]
        best_params["rd_rate"] = config["rd_rate"]
    except:
        best_params["rd_start"] = -1
        best_params["rd_end"] = -1
        best_params["rd_rate"] = 1
                
    total_test_reps = 0
    det_path_computed = False
    cost_record_ij = []
    stoch_outputs_test = []
    unique_seeds = []
    real_hosp = hosp_benchmark
    while len(stoch_outputs_test) < n_replicas_test:
        chunksize = 4 if mp_pool is None else 4 * mp_pool._processes
        total_test_reps += chunksize
        if unique_seeds_ori == []:
            # no input seeds
            seed_shift_var=10_00000 + total_test_reps
            crn_input = None
            chunkinput = chunksize
        else:
            seed_shift_var = 0
            crn_input = unique_seeds_ori[total_test_reps-chunksize:total_test_reps]
            if len(crn_input) == 0:
                # if the given seeds are run out, need to generate new seeds
                chunkinput = chunksize
                crn_input = None
                seed_shift_var = unique_seeds_ori[-1] + 1 + total_test_reps
            else:
                chunkinput = len(crn_input)
                
        rsq_thr = config["rsq_threshold"]
        change_dates = [dt.datetime(2020, 3, 23, 0, 0), dt.datetime(2020, 5, 20, 0, 0), dt.datetime(2020, 6, 25, 0, 0), dt.datetime(2020, 8, 19, 0, 0), dt.datetime(2020, 10, 7, 0, 0)]
        stoch_next = []
        seed_next = []
        for k in range(0, len(change_dates)):
            if k == 0:
                best_params['particle_filtering'] = True
                best_params['end_date'] = change_dates[0]
                best_params['start_date'] = instance.start_date
                seed_next = crn_input
                shift = seed_shift_var
                replica_no = chunksize
            else:
                best_params['end_date'] = change_dates[k]
                best_params['start_date'] = change_dates[k - 1] + dt.timedelta(days=1)
                shift = 0
                replica_no = len(seed_next)
                
            ###############Run stochastic simulations with known distribution
            out_sample_configs = stoch_simulation_iterator_filter(instance,
                                                                  best_policy,
                                                                  obj_func,
                                                                  interventions_train,
                                                                  ex_res=stoch_next,
                                                                  seed_shift=shift,
                                                                  crn_seeds=seed_next,
                                                                  n_replicas=replica_no,
                                                                  det_sample_path=False,
                                                                  **best_params)
            out_sample_outputs = simulate_p(mp_pool, out_sample_configs)
           
            T = 1 + (best_params['end_date'] - instance.start_date).days
            seed_temp = []
            r_sq_path = []
            stoch_outputs_temp = []
            ids = []
            for sample_ij in out_sample_outputs:
                sim_j, cost_j, policy_j, seed_j, kwargs_j = sample_ij
                stoch_outputs_temp.append(sample_ij)
                seed_temp.append(seed_j)
                IH_j = np.sum(sim_j['IH'][0:T] + sim_j['ICU'][0:T], axis = (1,2))
                r_sq_path.append(1 - sum(np.square(real_hosp[0:T]- IH_j))/sum(np.square(real_hosp[0:T] - np.mean(real_hosp[0:T]))))
            
            candidate_ids = [i for i,x in enumerate(r_sq_path) if x > rsq_thr]
            r_sq_path = [r_sq_path[i] for i in candidate_ids]
            if len(r_sq_path) > 0:
                ids = candidate_ids
                stoch_next = [stoch_outputs_temp[i] for i in ids] 
                seed_next = [seed_temp[i] for i in ids] 
            else:
                break
            
        if len(ids) > 0:
            best_params['end_date'] = instance.end_date
            best_params['start_date'] = change_dates[-1] + dt.timedelta(days=1)
            T = 1 + (best_params['end_date'] - best_params['start_date']).days
            ###############Run stochastic simulations with known distribution
            out_sample_configs = stoch_simulation_iterator_filter(instance,
                                                              best_policy,
                                                              obj_func,
                                                              interventions_train,
                                                              ex_res=stoch_next,
                                                              seed_shift=0,
                                                              crn_seeds=seed_next,
                                                              n_replicas=len(ids),
                                                              det_sample_path=False,
                                                              **best_params)
            out_sample_outputs = simulate_p(mp_pool, out_sample_configs)   
            
            for sample_ij in out_sample_outputs:
                sim_j, cost_j, policy_j, seed_j, kwargs_j = sample_ij
                cost_record_ij.append(kwargs_j["cost_info"])
                stoch_outputs_test.append(sample_ij)
                unique_seeds.append(seed_j)
                print(seed_j," out of ",total_test_reps)
                if seed_j == -1:
                    det_path_computed = True
                if len(stoch_outputs_test) == n_replicas_test:
                    break
                
    stoch_outputs_test[0] = all_outputs[best_id]    
    assert len(np.unique(unique_seeds)) == n_replicas_test
    bpStr = str(best_policy)
    if bpStr not in cost_record.keys():
        cost_record[str(best_policy)] = cost_record_ij
    logger.info(f'Got {len(stoch_outputs_test)} replicas sampling {total_test_reps}')
    stoch_replicas = [rep_i[0] for rep_i in stoch_outputs_test]
    stoch_costs = [rep_i[1] for rep_i in stoch_outputs_test]
    infeasible_replicas = np.sum([
        np.any(stoch_replicas[rep_i]['IH'].sum(axis=(1, 2)) > instance.hosp_beds)
        for rep_i in range(len(stoch_replicas))
    ])
    IH_feasible = infeasible_replicas <= int(config['chance_constraint_epsilon'] * n_replicas_train)
    expected_cost = np.mean(stoch_costs)
    logger.info(f'Optimized policy: {str(best_policy)}')
    logger.info(f'Cost: {expected_cost}')
    logger.info(f'Inf scenarios: {infeasible_replicas} out of {len(stoch_costs)}')
    print_profiling_log(logger)
    # Save solution
    instance_name = instance_name if instance_name is not None else f'output_{instance.city}.p'
    file_path = output_path / f'{instance_name}.p'
    if file_path.is_file():
        file_path = output_path / f'{instance_name}_{str(dt.datetime.now())}.p'
    crns_out = np.array(crn_seeds)
    unique_out = np.array(unique_seeds)
    with open(str(file_path), 'wb') as outfile:
        pickle.dump(
            (instance, interventions_train, best_params, best_policy, best_sim, stoch_replicas, config, cost_record, (crns_out[crns_out >= 0], unique_out[unique_out >= 0])),
            outfile, pickle.HIGHEST_PROTOCOL)
    
    return stoch_replicas, best_policy, file_path


