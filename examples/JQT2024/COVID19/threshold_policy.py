'''
    Module to compute threshold type policies
'''
import pickle
import numpy as np
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
from interventions import create_intLevel, form_interventions
from itertools import product
from SEIYAHRD_sim import simulate, hosp_based_policy, fix_policy, simulate_p
from policies import build_multi_tier_policy_candidates, build_ACS_policy_candidates, MultiTierPolicy, MultiTierPolicy_ACS
#from reporting.plotting import plot_stoch_simulations
from COVID19 import config, logger, output_path
from covid_utils import profile_log, print_profiling_log

datetime_formater = '%Y-%m-%d %H:%M:%S'
date_formater = '%Y-%m-%d'

def run_multi_calendar(instance, tiers, interventions):
    '''
        Runs the calendar to fix decisions regarding school closures,
        cocooning and social distance. If decisions are not fixed,
        creates a list of feasible interventions for every time period.
        Previous two options, lock-down or relaxation, have been expanded to a policy tier
    '''
    # Local variables
    T = instance.T
    cal = instance.cal
    # Run callendar and set what's already decided
    int_dict = {(i.SC, i.CO, i.SD): ix for ix, i in enumerate(interventions)}
    z_ini = np.array([None] * (T - 1))
    SD_state = np.array([None] * (T - 1))
    feasible_interventions = []
    
    for t in range(T - 1):
        d = cal.calendar[t]
        
        # if it is marked for school closure in the instance
        if cal.schools_closed[t]:
            sc = 1
        else:
            sc = 0  # TODO: We might need to add the parameter again if SC is part of the optimization
        
        transmission_reduction = cal.fixed_transmission_reduction[t]
        cocooning_hist = cal.fixed_cocooning[t]
        if transmission_reduction is not None:
            # when there is a transmission reduction
            z_ini[t] = int_dict[sc, cocooning_hist, transmission_reduction]
            feasIt = {}
            # find the "do-nothing" option in the tier list
            maxTier = 0
            maxTierIter = 0
            for iTier in range(len(tiers)):
                feasIt[iTier] = z_ini[t]
                # SD_state equals to the maxTierIter
                if (transmission_reduction >=
                        tiers[iTier]['transmission_reduction']) and (tiers[iTier]['transmission_reduction'] > maxTier):
                    maxTier = tiers[iTier]['transmission_reduction']
                    maxTierIter = iTier
            SD_state[t] = maxTierIter
            feasible_interventions.append(feasIt)
            
            #fix 14 days into the future
        
        elif transmission_reduction is None and d <= instance.last_date_interventions:
            # Assumption: if High, schools need to be closed
            z_ini[t] = None
            feasIt = {}
            for iTer in range(len(tiers)):
                feasIt[iTer] = int_dict[max(tiers[iTer]['school_closure'], sc
                                            ), tiers[iTer]['cocooning'], tiers[iTer]['transmission_reduction']]
            feasible_interventions.append(feasIt)
            SD_state[t] = None
        else:
            z_ini[t] = None
            feasIt = {}
            for iTer in range(len(tiers)):
                feasIt[iTer] = int_dict([0, 0, 0])
            feasible_interventions.append(feasIt)
            SD_state[t] = None
    
    return z_ini, SD_state, feasible_interventions


def policy_multi_iterator(instance, tiers, obj_fun, interventions, policy_class='constant', fixed_policy=None, 
                          fo_tiers=None, redLimit=1000, after_tiers=[0,1,2,3,4], policy_field="IYIH", policy_ub=None,
                          acs_set=False, acs_bounds=(0,0), acs_time_bounds=(0,0), acs_lead_time=0, acs_Q=0):
    '''
        Creates an iterator of the candidate thresholds for each tier. The iterator will be used
        to map the simulator in parallel using the helper function simulate_p on the
        simulation module.

        Args:
        instance (module): a python module with all the required input
        tier (list of dict): list of candidate tiers
        policy_class (str): class of policy to optimize. Options are:
            "constant": optimizes one parameter and safety threshold
            "step": optimizes three parameters (first threshold, second threshold, and last month in
                    which it changes to the second threshold) and safety threshold
        fixed_policy (dict): if provided, no search is excecuted and the iterator yields one policy.
            Signature of fixed_policy = { policy_class: "a class listed above",
                                          vals: [val1, val2, val3]
                                        }
    '''
    first_day_month_index = defaultdict(int)
    first_day_month_index.update({(d.month, d.year): t for t, d in enumerate(instance.cal.calendar) if (d.day == 1)})
    z_ini, SD_state, feasible_interventions = run_multi_calendar(instance, tiers, interventions)
    kwargs = {
        'hosp_beds': instance.hosp_beds,
        'tiers': tiers,
        'feasible_interventions': feasible_interventions,
        'lambda_star': instance.lambda_star,
        'infeasible_penalty': config['infeasible_penalty'],
        'over_capacity_cost': config['over_capacity_cost'],
        'obj_over_included': config['obj_over_included'],
        'sim_method': config['sim_method'],
        'active_intervention': config['active_intervention'],
        'policy_field': policy_field,
        'fo_tiers': fo_tiers,
        'changed_tiers': False,
        'redLimit': redLimit,
        'after_tiers': after_tiers,
        'extra_capacity_cost': config['extra_capacity_cost'],
        'capacity_setup_cost': config['capacity_setup_cost'],
        "icu_trigger": config['icu_trigger'],
        "icu_capacity_cost": config['icu_capacity_cost'],
        "acs_policy_field": config['acs_policy_field']
    }
    
    try:
        start_date = dt.datetime.strptime(config["rd_start"], datetime_formater)
        kwargs["rd_start"] = instance.cal.calendar_ix[start_date]
        end_date = dt.datetime.strptime(config["rd_end"], datetime_formater)
        kwargs["rd_end"] = instance.cal.calendar_ix[end_date]
        kwargs["rd_rate"] = config["rd_rate"]
    except:
        kwargs["rd_start"] = -1
        kwargs["rd_end"] = -1
        kwargs["rd_rate"] = 1
    
    if fixed_policy is None:
        if acs_set:
            for thrs,acs_thrs,acs_length in build_ACS_policy_candidates(instance, tiers, acs_bounds, acs_time_bounds,
                                                           threshold_type=policy_class, lambda_start=policy_ub):
                mt_policy = MultiTierPolicy_ACS(instance, tiers, thrs, acs_thrs, acs_length, acs_lead_time, acs_Q)
                mt_policy.set_tier_history(SD_state.copy())
                mt_policy.set_intervention_history(z_ini.copy())
                yield instance, mt_policy, obj_fun, interventions, -1, kwargs
        else:
            for thrs in build_multi_tier_policy_candidates(instance, tiers, threshold_type=policy_class, lambda_start=policy_ub):
                mt_policy = MultiTierPolicy(instance, tiers, thrs)
                mt_policy.set_tier_history(SD_state.copy())
                mt_policy.set_intervention_history(z_ini.copy())
                yield instance, mt_policy, obj_fun, interventions, -1, kwargs
    else:
        fixed_policy.set_tier_history(SD_state.copy())
        fixed_policy.set_intervention_history(z_ini.copy())
        yield instance, fixed_policy, obj_fun, interventions, -1, kwargs

def stoch_simulation_iterator(instance,
                              policy,
                              obj_func,
                              interventions,
                              det_sample_path=True,
                              crn_seeds=None,
                              seed_shift=0,
                              n_replicas=300,
                              **kwargs):
    '''
        Creates an iterator for different replicas, changing the seed for the random stream.
        The iterator will be used to map the simulator in parallel using the helper function
        simulate_p on the simulation module.

        Args:
            n_replicas (int): number of stochastic simulations
            instance (module): python module with input data
            interventions (list): list of all interventions to be consider in the horizon
            sd_levels (dict): a map from lock-down/relaxation to transmission reduction (kappa)
            cocooning (float): level of transmission reduction, [0,1], for high risk and 65+ groups
            school_closure (int): 1 schools are closed, 0 schools are open unless is fixed otherwise.
            params_policy (dict): paramters of the policy to be simulated. The signature is this
                dictionary comes from kwargs built in the function policy_input_iterator.
    '''
    reps = n_replicas + 1 if det_sample_path else n_replicas
    seeds = []
    if det_sample_path:
        seeds.append(-1)
    seeds.extend(crn_seeds if crn_seeds is not None else range(n_replicas))
    z_ini, SD_state, feasible_interventions = run_multi_calendar(instance, policy.tiers, interventions)
    for rep_i in range(reps):
        r_seed = seeds[rep_i] + (seed_shift if seeds[rep_i] >= 0 else 0)
        policy_copy = policy.deep_copy()
        policy_copy.set_tier_history(SD_state.copy())
        policy_copy.set_intervention_history(z_ini.copy())
        yield instance, policy_copy, obj_func, interventions, r_seed, kwargs


# policy_multi_iterator_filter() and stoch_simulation_iterator_filter() are used for downsampling

def policy_multi_iterator_filter(instance, tiers, obj_fun, interventions, policy_class='constant', fixed_policy=None, 
                          fo_tiers=None, redLimit=1000, after_tiers=[0,1,2,3,4], policy_field="IYIH", policy_ub=None,
                          acs_set=False, acs_bounds=(0,0), acs_time_bounds=(0,0), acs_lead_time=0, acs_Q=0):
    '''
        Creates an iterator of the candidate thresholds for each tier. The iterator will be used
        to map the simulator in parallel using the helper function simulate_p on the
        simulation module.

        Args:
        instance (module): a python module with all the required input
        tier (list of dict): list of candidate tiers
        policy_class (str): class of policy to optimize. Options are:
            "constant": optimizes one parameter and safety threshold
            "step": optimizes three parameters (first threshold, second threshold, and last month in
                    which it changes to the second threshold) and safety threshold
        fixed_policy (dict): if provided, no search is excecuted and the iterator yields one policy.
            Signature of fixed_policy = { policy_class: "a class listed above",
                                          vals: [val1, val2, val3]
                                        }
    '''
    first_day_month_index = defaultdict(int)
    first_day_month_index.update({(d.month, d.year): t for t, d in enumerate(instance.cal.calendar) if (d.day == 1)})
    z_ini, SD_state, feasible_interventions = run_multi_calendar(instance, tiers, interventions)
    kwargs = {
        'hosp_beds': instance.hosp_beds,
        'tiers': tiers,
        'feasible_interventions': feasible_interventions,
        'lambda_star': instance.lambda_star,
        'infeasible_penalty': config['infeasible_penalty'],
        'over_capacity_cost': config['over_capacity_cost'],
        'obj_over_included': config['obj_over_included'],
        'sim_method': config['sim_method'],
        'active_intervention': config['active_intervention'],
        'policy_field': policy_field,
        'fo_tiers': fo_tiers,
        'changed_tiers': False,
        'redLimit': redLimit,
        'after_tiers': after_tiers,
        'extra_capacity_cost': config['extra_capacity_cost'],
        'capacity_setup_cost': config['capacity_setup_cost'],
        "icu_trigger": config['icu_trigger'],
        "icu_capacity_cost": config['icu_capacity_cost'],
        "acs_policy_field": config['acs_policy_field'],
        "particle_filtering": config['particle_filtering']
    }
    
    try:
        start_date = dt.datetime.strptime(config["rd_start"], datetime_formater)
        kwargs["rd_start"] = instance.cal.calendar_ix[start_date]
        end_date = dt.datetime.strptime(config["rd_end"], datetime_formater)
        kwargs["rd_end"] = instance.cal.calendar_ix[end_date]
        kwargs["rd_rate"] = config["rd_rate"]
    except:
        kwargs["rd_start"] = -1
        kwargs["rd_end"] = -1
        kwargs["rd_rate"] = 1
    
    if fixed_policy is None:
        if acs_set:
            for thrs,acs_thrs,acs_length in build_ACS_policy_candidates(instance, tiers, acs_bounds, acs_time_bounds,
                                                           threshold_type=policy_class, lambda_start=policy_ub):
                mt_policy = MultiTierPolicy_ACS(instance, tiers, thrs, acs_thrs, acs_length, acs_lead_time, acs_Q)
                mt_policy.set_tier_history(SD_state.copy())
                mt_policy.set_intervention_history(z_ini.copy())
                yield [], instance, mt_policy, obj_fun, interventions, -1, kwargs
        else:
            for thrs in build_multi_tier_policy_candidates(instance, tiers, threshold_type=policy_class, lambda_start=policy_ub):
                mt_policy = MultiTierPolicy(instance, tiers, thrs)
                mt_policy.set_tier_history(SD_state.copy())
                mt_policy.set_intervention_history(z_ini.copy())
                yield [], instance, mt_policy, obj_fun, interventions, -1, kwargs
    else:
        fixed_policy.set_tier_history(SD_state.copy())
        fixed_policy.set_intervention_history(z_ini.copy())
        yield [], instance, fixed_policy, obj_fun, interventions, -1, kwargs
        
        
def stoch_simulation_iterator_filter(instance,
                              policy,
                              obj_func,
                              interventions,
                              ex_res,
                              det_sample_path=True,
                              crn_seeds=None,
                              seed_shift=0,
                              n_replicas=300,
                              **kwargs):
    '''
        Creates an iterator for different replicas, changing the seed for the random stream.
        The iterator will be used to map the simulator in parallel using the helper function
        simulate_p on the simulation module.

        Args:
            n_replicas (int): number of stochastic simulations
            instance (module): python module with input data
            interventions (list): list of all interventions to be consider in the horizon
            sd_levels (dict): a map from lock-down/relaxation to transmission reduction (kappa)
            cocooning (float): level of transmission reduction, [0,1], for high risk and 65+ groups
            school_closure (int): 1 schools are closed, 0 schools are open unless is fixed otherwise.
            params_policy (dict): paramters of the policy to be simulated. The signature is this
                dictionary comes from kwargs built in the function policy_input_iterator.
    '''
    reps = n_replicas + 1 if det_sample_path else n_replicas
    seeds = []
    if det_sample_path:
        seeds.append(-1)
    seeds.extend(crn_seeds if crn_seeds is not None else range(n_replicas))
    z_ini, SD_state, feasible_interventions = run_multi_calendar(instance, policy.tiers, interventions)
    for rep_i in range(reps):
        r_seed = seeds[rep_i] + (seed_shift if seeds[rep_i] >= 0 else 0)
        if (kwargs['start_date'] != instance.start_date):
            ex_sim = ex_res[rep_i] 
            sim_result, cost_j, policy_j, seed_j, kwargs_j = ex_sim
        else:
            ex_sim = []
            policy_j = policy.deep_copy()
            policy_j.set_tier_history(SD_state.copy())
            policy_j.set_intervention_history(z_ini.copy())
        yield ex_sim, instance, policy_j, obj_func, interventions, r_seed, kwargs

