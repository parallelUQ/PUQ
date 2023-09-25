import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
import multiprocessing as mp
from utils import parse_arguments
from InterventionsMIP import load_config_file, logger, change_paths
from pathlib import Path
from pipelinemultitier import read_hosp
from interventions import create_intLevel, form_interventions
from itertools import product
from SEIYAHRD_sim import simulate, hosp_based_policy, fix_policy, simulate_p
from policies import build_multi_tier_policy_candidates, build_ACS_policy_candidates, MultiTierPolicy, MultiTierPolicy_ACS
from InterventionsMIP import config, logger, output_path
from utils import profile_log, print_profiling_log
from threshold_policy import policy_multi_iterator, run_multi_calendar, stoch_simulation_iterator
from objective_functions import multi_tier_objective
from instances import load_instance, load_tiers, load_seeds
from policies import MultiTierPolicy as MTP
from itertools import product
import warnings
from scipy.optimize import least_squares
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import math 
def deterministic_path(instance, 
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

    fixed_TR = list(filter(None, instance.cal.fixed_transmission_reduction))
    tier_TR = [item['transmission_reduction'] for item in tiers]
    uniquePS = sorted(np.unique(np.append(fixed_TR, np.unique(tier_TR))))
    sc_levels = np.unique([tier['school_closure'] for tier in tiers] + [0, 1])
    
    fixed_CO = list(filter(None, instance.cal.fixed_cocooning))
    tier_CO = np.unique([tier['cocooning'] for tier in tiers])
    uniqueCO = sorted(np.unique(np.append(fixed_CO, np.unique(tier_CO))))

    intervention_levels = create_intLevel(sc_levels, uniqueCO, uniquePS)
    interventions_train = form_interventions(intervention_levels, instance.epi, instance.N)
    
    # Build an iterator of all the candidates to be simulated by simulate_p
    sim_configs = policy_multi_iterator(instance,
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
    
    if len(all_outputs) == 1:
        # Skip search if there is only one candidate
        sim_output, cost, best_policy, seed_0, kwargs_out = all_outputs[0]
        
    return sim_output

def residual_error(x_beta, **kwargs):
    change_dates = kwargs['change_dates']
    instance = kwargs['instance']
    tiers = kwargs['tiers']
    hosp_ad = kwargs['hosp_ad']
    real_icu = kwargs['real_icu']
    
    #Change the transmission reduction and cocconing accordingly
    beta = [x_beta[0], x_beta[1], x_beta[2], x_beta[3], x_beta[4]]
    cocoon = [0, x_beta[1], x_beta[1], x_beta[3], x_beta[4]]

    tr_reduc = []
    date_list = []
    cocoon_reduc = []
    for idx in range(len(change_dates[:-1])):
        tr_reduc.extend([beta[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
        date_list.extend([str(change_dates[idx] + dt.timedelta(days=x)) for x in range((change_dates[idx + 1] - change_dates[idx]).days)])
        cocoon_reduc.extend([cocoon[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
        
    d = {'date': pd.to_datetime(date_list), 'transmission_reduction': tr_reduc}
    df_transmission = pd.DataFrame(data=d)
    transmission_reduction = [(d, tr) for (d, tr) in zip(df_transmission['date'], df_transmission['transmission_reduction'])]
    instance.cal.load_fixed_transmission_reduction(transmission_reduction, present_date=instance.end_date)
    
    d = {'date': pd.to_datetime(date_list), 'cocooning': cocoon_reduc}
    df_cocooning = pd.DataFrame(data=d)
    cocooning = [(d, c) for (d, c) in zip(df_cocooning['date'], df_cocooning['cocooning'])]
    instance.cal.load_fixed_cocooning(cocooning, present_date=instance.end_date)
    #############
    
    train_seeds, test_seeds = load_seeds(instance.city,'seed.p')
    #tiers = load_tiers(instance.city, tier_file_name=args.t)
    
    # TODO Read command line args for n_proc for better integration with crunch
    n_proc = 1
    
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = 1
    n_replicas_test = 1
    
    # Create the pool (Note: pool needs to be created only once to run on a cluster)
    mp_pool = mp.Pool(n_proc) if n_proc > 1 else None
    
    # check if the "do-nothing" / 'Stage 1 option is in the tiers. If not, add it
    originInt = {
        "name": "Stage 1",
        "transmission_reduction": 0,
        "cocooning": 0,
        "school_closure": 0,
        "min_enforcing_time": 1,
        "daily_cost": 0,
        "color": 'white'
        }
   
    if tiers.tier_type == 'constant':
        originInt["candidate_thresholds"] = [-1]  # Means that there is no lower bound
    elif tiers.tier_type == 'step':
        originInt["candidate_thresholds"] = [[-1], [-0.5]]
    
    if not (originInt in tiers.tier):
        tiers.tier.insert(0, originInt)
    
    given_threshold = eval('[-1,0,5,20,70]')
    given_date = None
    # if a threshold/threshold+stepping date is given, then it carries out a specific task
    # if not, then search for a policy
    selected_policy = None
    if tiers.tier_type == 'constant':
        if given_threshold is not None:
            selected_policy = MTP.constant_policy(instance, tiers.tier, given_threshold)
    elif tiers.tier_type == 'step':
        if (given_threshold is not None) and (given_date is not None):
            selected_policy = MTP.step_policy(instance, tiers.tier, given_threshold, given_date)
    
    task_str = str(selected_policy) if selected_policy is not None else f'opt{len(tiers.tier)}'
    instance_name = 'det_path'
    
    # read in the policy upper bound
    policy_ub = None

    # Set alphas
    instance.epi.alpha1 = x_beta[5]
    instance.epi.alpha2 = x_beta[6]
    
    sim_output = deterministic_path(instance=instance,
                                    tiers=tiers.tier,
                                    obj_func=multi_tier_objective,
                                    n_replicas_train=n_replicas_train,
                                    n_replicas_test=n_replicas_test,
                                    instance_name=instance_name,
                                    policy_class=tiers.tier_type,
                                    policy=selected_policy,
                                    mp_pool=mp_pool,
                                    crn_seeds=train_seeds,
                                    unique_seeds_ori=test_seeds,
                                    forcedOut_tiers=eval('[]'),
                                    redLimit=100000,
                                    after_tiers=eval('[0,1,2,3,4]'),
                                    policy_field='IYIH',
                                    policy_ub=policy_ub)

 
    if instance.city == 'austin':
        hosp_benchmark = None
        real_hosp = [a_i - b_i for a_i, b_i in zip(instance.cal.real_hosp, real_icu)] 
        hosp_benchmark = [sim_output['IH'][t].sum() for t in range(len(instance.cal.real_hosp))]
        residual_error_IH = [a_i - b_i for a_i, b_i in zip(real_hosp, hosp_benchmark)]
        
        icu_benchmark = [sim_output['ICU'][t].sum() for t in range(len(instance.cal.real_hosp))]
        w_icu = 1.5
        residual_error_ICU = [a_i - b_i for a_i, b_i in zip(real_icu, icu_benchmark)]
        residual_error_ICU = [element * w_icu for element in residual_error_ICU]
        residual_error_IH.extend(residual_error_ICU)
    
        w_iyih = 7.3*(1 - 0.10896) + 9.9*0.10896
        daily_ad_benchmark = [sim_output['ToIHT'][t].sum() for t in range(len(instance.cal.real_hosp) - 1)] 
        residual_error_IYIH = [a_i - b_i for a_i, b_i in zip(hosp_ad, daily_ad_benchmark)]
        residual_error_IYIH = [element * w_iyih for element in residual_error_IYIH]
        residual_error_IH.extend(residual_error_IYIH)
        
    elif instance.city == 'houston':
        hosp_benchmark = None
        real_hosp = [a_i - b_i for a_i, b_i in zip(instance.cal.real_hosp, real_icu)] 
        hosp_benchmark = [sim_output['IH'][t].sum() for t in range(44, len(instance.cal.real_hosp))]
        residual_error_IH = [a_i - b_i for a_i, b_i in zip(real_hosp[44:], hosp_benchmark)]
 
        icu_benchmark = [sim_output['ICU'][t].sum() for t in range(44, len(instance.cal.real_hosp))]
        w_icu = 1.5
        residual_error_ICU = [a_i - b_i for a_i, b_i in zip(real_icu[44:], icu_benchmark)]
        residual_error_ICU = [element * w_icu for element in residual_error_ICU]
        residual_error_IH.extend(residual_error_ICU)
    
    return residual_error_IH 
    
def least_squares_fit(initial_guess, kwargs):
    # Function that runs the least squares fit
    result = least_squares(residual_error, initial_guess, bounds = ([0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]), method='trf', kwargs = kwargs)
    return result 

    
def run_fit(instance,
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
            policy_ub=None,
            method="lsq"):
    
    if instance.city == 'austin':
        start_date = dt.datetime(2020,2,28)
        daily_admission_file_path = instance.path_to_data  / "austin_hosp_ad_lsq.csv"
        hosp_ad = read_hosp(daily_admission_file_path, start_date, "admits")
        
        daily_icu_file_path = instance.path_to_data  / "austin_real_icu_lsq.csv"
        real_icu = read_hosp(daily_icu_file_path, start_date)  

        #time blocks
        change_dates = [dt.date(2020, 2, 15), dt.date(2020, 3, 24), dt.date(2020, 5, 21), dt.date(2020, 6, 26), dt.date(2020, 8, 20), dt.date(2020, 10, 8)] 

        #initial guess
        x = np.array([0, 0.74, 0.8, 0.8, 0.8, 0.8, 0.8])        
   
    elif instance.city == 'houston':
        hosp_ad = None
        daily_icu_file_path = instance.path_to_data  / "houston_real_icu_lsq.csv"
        start_date = dt.datetime(2020,2,19)
        real_icu = read_hosp(daily_icu_file_path, start_date)   
        
        #time blocks
        change_dates = [dt.date(2020, 2, 15), dt.date(2020, 3, 24), dt.date(2020, 5, 21), dt.date(2020, 6, 26), dt.date(2020, 8, 20), dt.date(2020, 10, 8)]
    
        #initial guess
        x = np.array([0, 0.74, 0.8, 0.8, 0.8, 0.8, 0.8])

    kwargs  = {'change_dates' : change_dates,
               'instance' : instance,
               'tiers' : tiers,
               'hosp_ad': hosp_ad,
               'real_icu': real_icu
               }
    
    ########## ########## ##########
    #Run least squares
    res = least_squares_fit(x, kwargs)
    SSE = res.cost
    ########## ########## ##########
    
    #Get variable value
    opt_tr_reduction = res.x
    contact_reduction = opt_tr_reduction[0:5]
    cocoon = [0, opt_tr_reduction[1], opt_tr_reduction[1], opt_tr_reduction[3], opt_tr_reduction[4]]
    betas = instance.epi.beta*(1 - (contact_reduction))
    end_date = []
    for idx in range(len(change_dates[1:])):
        end_date.append(str(change_dates[1:][idx] - dt.timedelta(days=1)))
    
    print('beta_0:', instance.epi.beta)   
    print('SSE:', SSE)   
    table = pd.DataFrame({'start_date': change_dates[:-1], 'end_date': end_date, 'contact_reduction': contact_reduction, 'beta': betas, 'cocoon': cocoon})
    print(table)
    
    print('alpha1=', opt_tr_reduction[5])
    print('alpha2=', opt_tr_reduction[6])

    tr_reduc = []
    date_list = []
    cocoon_reduc = []
    for idx in range(len(change_dates[:-1])):
        tr_reduc.extend([contact_reduction[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
        date_list.extend([str(change_dates[idx] + dt.timedelta(days=x)) for x in range((change_dates[idx + 1] - change_dates[idx]).days)])
        cocoon_reduc.extend([cocoon[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
    
    d = {'date': pd.to_datetime(date_list), 'transmission_reduction': tr_reduc, 'cocooning': cocoon_reduc}
    df_transmission = pd.DataFrame(data=d)

    return df_transmission
 