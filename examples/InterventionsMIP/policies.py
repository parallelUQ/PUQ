'''
This module includes different policies that are simulated
'''
import json
import numpy as np
from InterventionsMIP import config
from itertools import product

CONSTANT_TR = 'constant'
STEP_TR = 'step'
THRESHOLD_TYPES = [CONSTANT_TR, STEP_TR]


def build_multi_tier_policy_candidates(instance, tiers, threshold_type='constant', lambda_start=None):
    assert len(tiers) >= 2, 'At least two tiers need to be defined'
    threshold_candidates = []
    if threshold_type == CONSTANT_TR:
        gz = config['grid_size']
        # lambda_start is given by the field pub; if it is none, then we use the square root staffing rule
        if lambda_start is None:
            if np.size(instance.epi.eq_mu) == 1:
                lambda_start = int(np.floor(instance.epi.eq_mu * instance.lambda_star))
            else:
                lambda_start = int(np.floor(np.max(instance.epi.eq_mu) * instance.lambda_star))
        params_trials = []
        for tier in tiers:
            if 'candidate_thresholds' in tier and isinstance(tier['candidate_thresholds'], list):
                params_trials.append(tier['candidate_thresholds'])
            else:
                candidates = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
                params_trials.append(np.unique(candidates))
        for policy_params in product(*params_trials):
            is_valid = True
            for p_ix in range(len(policy_params) - 1):
                if policy_params[p_ix] >= policy_params[p_ix + 1]:
                    is_valid = False
                    break
            if is_valid:
                T = len(instance.cal)
                lockdown_thresholds = [[policy_params[i]] * T for i in range(len(policy_params))]
                threshold_candidates.append(lockdown_thresholds)
        return threshold_candidates
    elif threshold_type == STEP_TR:
        # TODO: we need to set a time and two levels
        gz = config['grid_size']
        lambda_start = int(np.floor(instance.epi.eq_mu * instance.lambda_star))
        params_trials = []
        for tier in tiers:
            if tier['candidate_thresholds'] is not None:
                # construct the trial parameters according to the candidate threshold
                # the candidate threshold should be a list of 2 lists
                if isinstance(tier['candidate_thresholds'][0], list):
                    candidates1 = tier['candidate_thresholds'][0]
                else:
                    candidates1 = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
                if isinstance(tier['candidate_thresholds'][1], list):
                    candidates2 = tier['candidate_thresholds'][1]
                else:
                    candidates2 = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
            else:
                candidates1 = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
                candidates2 = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
            params_trials.append([(t1, t2) for t1 in candidates1 for t2 in candidates2 if t1 <= t2])
        # obtain the possible stepping time points, limited to the start of months
        T_trials = instance.cal.month_starts
        for policy_params in product(*params_trials):
            is_valid = True
            for p_ix in range(len(policy_params) - 1):
                if (policy_params[p_ix][0] >= policy_params[p_ix + 1][0]) \
                        or (policy_params[p_ix][1] >= policy_params[p_ix + 1][1]):
                    is_valid = False
                    break
            if is_valid:
                T = len(instance.cal)
                for tChange in T_trials:
                    lockdown_thresholds = [[policy_params[i][0]] * tChange + [policy_params[i][1]] * (T - tChange)
                                           for i in range(len(policy_params))]
                    threshold_candidates.append(lockdown_thresholds)
        return threshold_candidates
    else:
        raise NotImplementedError

def build_ACS_policy_candidates(instance, tiers, acs_bounds, acs_time_bounds, threshold_type='constant', lambda_start=None):
    assert len(tiers) >= 2, 'At least two tiers need to be defined'
    threshold_candidates = []
    if threshold_type == CONSTANT_TR:
        gz = config['grid_size']
        if lambda_start is None:
            if np.size(instance.epi.eq_mu) == 1:
                lambda_start = int(np.floor(instance.epi.eq_mu * instance.lambda_star))
            else:
                lambda_start = int(np.floor(np.max(instance.epi.eq_mu) * instance.lambda_star))
        params_trials = []
        for tier in tiers:
            if 'candidate_thresholds' in tier and isinstance(tier['candidate_thresholds'], list):
                params_trials.append(tier['candidate_thresholds'])
            else:
                candidates = [gz * i for i in range(0, int(lambda_start / gz) + 1)] + [lambda_start]
                params_trials.append(candidates)
        # append the acs_trigger and acs_length
        acs_trigger_candidates = np.unique([gz * i for i in range(int(acs_bounds[0] / gz), int(acs_bounds[1] / gz) + 1)] + [acs_bounds[1]])
        acs_time_candidates = np.unique([gz * i for i in range(int(acs_time_bounds[0] / gz), int(acs_time_bounds[1] / gz) + 1)] + [acs_time_bounds[1]])
        params_trials.append(acs_trigger_candidates)
        params_trials.append(acs_time_candidates)
        
        for policy_params in product(*params_trials):
            is_valid = True
            for p_ix in range(len(policy_params) - 3):
                if policy_params[p_ix] >= policy_params[p_ix + 1]:
                    is_valid = False
                    break
            if is_valid:
                T = len(instance.cal)
                lockdown_thresholds = [[policy_params[i]] * T for i in range(len(policy_params) - 2)]
                output_trials = [lockdown_thresholds, policy_params[-2], policy_params[-1]]
                threshold_candidates.append(output_trials)
        return threshold_candidates
    else:
        raise NotImplementedError

class MultiTierPolicy():
    '''
        A multi-tier policy allows for multiple tiers of lock-downs.
        Attrs:
            tiers (list of dict): a list of the tiers characterized by a dictionary
                with the following entries:
                    {
                        "transmission_reduction": float [0,1)
                        "cocooning": float [0,1)
                        "school_closure": int {0,1}
                    }
            
            lockdown_thresholds (list of list): a list with the thresholds for every
                tier. The list must have n-1 elements if there are n tiers. Each threshold
                is a list of values for evert time step of simulation.
            threshold_type: functional form of the threshold (options are in THRESHOLD_TYPES)
    '''
    def __init__(self, instance, tiers, lockdown_thresholds):
        assert len(tiers) == len(lockdown_thresholds)
        self.tiers = tiers
        self.lockdown_thresholds = lockdown_thresholds
        self.lockdown_thresholds_ub = [lockdown_thresholds[i] for i in range(1, len(lockdown_thresholds))]
        self.lockdown_thresholds_ub.append([np.inf] * len(lockdown_thresholds[0]))
        
        self._n = len(self.tiers)
        self._tier_history = None
        self._intervention_history = None
        self._instance = instance
        
        self.red_counter = 0
    
    @classmethod
    def constant_policy(cls, instance, tiers, constant_thresholds):
        T = instance.T
        lockdown_thresholds = [[ct] * T for ct in constant_thresholds]
        return cls(instance, tiers, lockdown_thresholds)
    
    @classmethod
    def step_policy(cls, instance, tiers, constant_thresholds, change_date):
        lockdown_thresholds = []
        for tier_ix, tier in enumerate(tiers):
            tier_thres = []
            for t, d in enumerate(instance.cal.calendar):
                if d < change_date:
                    tier_thres.append(constant_thresholds[0][tier_ix])
                else:
                    tier_thres.append(constant_thresholds[1][tier_ix])
            lockdown_thresholds.append(tier_thres)
        return cls(instance, tiers, lockdown_thresholds)
    
    def deep_copy(self):
        p = MultiTierPolicy(self._instance, self.tiers, self.lockdown_thresholds)
        p.set_tier_history(self._tier_history_copy)
        p.set_intervention_history(self._intervention_history_copy)
        return p
    
    def set_tier_history(self, history):
        # Set history and saves a copy to reset
        self._tier_history = history.copy()
        self._tier_history_copy = history.copy()
    
    def set_intervention_history(self, history):
        # Set history and saves a copy to reset
        self._intervention_history = history.copy()
        self._intervention_history_copy = history.copy()
    
    def reset_history(self):
        # reset history so that a new simulation can be excecuted
        self.set_tier_history(self._tier_history_copy)
        self.set_intervention_history(self._intervention_history_copy)
    
    def compute_cost(self):
        return sum(self.tiers[i]['daily_cost'] for i in self._tier_history if i is not None and i in range(self._n))
    
    def get_tier_history(self):
        return self._tier_history
    
    def get_interventions_history(self):
        return self._intervention_history
    
    def __repr__(self):
        p_str = str([(self.tiers[i]['name'], self.lockdown_thresholds[i][0], self.lockdown_thresholds[i][-1])
                     for i in range(len(self.tiers))])
        p_str = p_str.replace(' ', '')
        p_str = p_str.replace(',', '_')
        p_str = p_str.replace("'", "")
        p_str = p_str.replace('[', '')
        p_str = p_str.replace('(', '')
        p_str = p_str.replace(']', '')
        p_str = p_str.replace(')', '')
        return p_str
    
    def __call__(self, t, criStat, IH, *args, **kwargs):
        '''
            Function that makes an instance of a policy a callable.
            Args:
                t (int): time period in the simulation
                z (object): deprecated, but maintained to avoid changes in the simulate function
                criStat (ndarray): the trigger statistics, previously daily admission, passed by the simulator
                IH (ndarray): hospitalizations admissions, passed by the simulator
                ** kwargs: additional parameters that are passed and are used elsewhere
        '''
        if self._tier_history[t] is not None:
            return self._intervention_history[t],kwargs
        
        # enforce the tiers out
        effective_threshold = {}
        effective_threshold_ub = {}
        for itier in range(len(self.tiers)):
            effective_threshold[itier] = self.lockdown_thresholds[itier][0]
            effective_threshold_ub[itier] = self.lockdown_thresholds_ub[itier][0]
        if kwargs['fo_tiers'] is None:
            effective_tiers = range(len(self.tiers))
        else:
            if not(kwargs['changed_tiers']):
                effective_tiers = [i for i in range(len(self.tiers)) if i not in kwargs['fo_tiers']]
            else:
                effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers']]
        for tier_ind in range(len(effective_tiers)):
            tier_ix = effective_tiers[tier_ind]
            if tier_ind != len(effective_tiers) - 1:
                effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
            else:
                effective_threshold_ub[tier_ix] = np.inf
        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - config['moving_avg_len'])
        criStat_total = criStat.sum((1, 2))
        criStat_avg = criStat_total[moving_avg_start:].mean()
        
        current_tier = self._tier_history[t - 1]
        valid_interventions_t = kwargs['feasible_interventions'][t]
        T = self._instance.T
        
        ## NEW
        new_tier = effective_tiers[[
            effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
            for tier_ix in effective_tiers
        ].index(True)]
        
        if new_tier == current_tier:
            t_end = np.minimum(t + 1, T)
            # if it is the first time turning red
            if new_tier == 4:
                # forced out of the red tier
                if self.red_counter >= kwargs["redLimit"]:
                    # if it is the first time turning red
                    if (not(kwargs['changed_tiers'])):
                        kwargs['changed_tiers'] = True
                    effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers'] and i != 4]
                    for tier_ind in range(len(effective_tiers)):
                        tier_ix = effective_tiers[tier_ind]
                        if tier_ind != len(effective_tiers) - 1:
                            effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
                        else:
                            effective_threshold_ub[tier_ix] = np.inf
                    new_tier = effective_tiers[[
                        effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
                        for tier_ix in effective_tiers
                    ].index(True)]
                    t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
        else:
            if new_tier < current_tier:
                IH_total = IH[-1].sum()
                if current_tier != 4:
                    assert_safety_trigger = IH_total < self._instance.hosp_beds * config['safety_threshold_frac']
                    new_tier = new_tier if assert_safety_trigger else current_tier
                    t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
                else:
                    # if it is the first time turning red
                    if (not(kwargs['changed_tiers'])):
                        kwargs['changed_tiers'] = True
                        effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers']]
                        for tier_ind in range(len(effective_tiers)):
                            tier_ix = effective_tiers[tier_ind]
                            if tier_ind != len(effective_tiers) - 1:
                                effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
                            else:
                                effective_threshold_ub[tier_ix] = np.inf
                        new_tier = effective_tiers[[
                            effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
                            for tier_ix in effective_tiers
                        ].index(True)]
                    t_delta = self.tiers[new_tier]['min_enforcing_time']
                t_end = np.minimum(t + t_delta, T)
            else:
                t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
        ## OLD
        # threshold_lb = self.lockdown_thresholds[current_tier][t] if current_tier > 0 else 0
        # threshold_ub = self.lockdown_thresholds[current_tier + 1][t] if current_tier + 1 < self._n - 1 else np.inf
        # if IYIH_avg > threshold_ub:  # bump to the next tier
        #     new_tier = np.minimum(current_tier + 1, self._n - 1)
        #     t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
        # elif IYIH_avg < threshold_lb:  # relax one tier, if safety trigger allows
        #     IH_total = IH[-1].sum()
        #     assert_safety_trigger = IH_total < self._instance.hosp_beds * config['safety_threshold_frac']
        #     new_tier = np.maximum(current_tier - 1, 0) if assert_safety_trigger else current_tier
        #     t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
        #     t_end = np.minimum(t + t_delta, T)
        # else:  # stay in same tier for one more time period
        #     new_tier = current_tier
        #     t_end = np.minimum(t + 1, T)
        self._intervention_history[t:t_end] = valid_interventions_t[new_tier]
        self._tier_history[t:t_end] = new_tier
        if new_tier == 4:
            self.red_counter += (t_end - t)
        else:
            self.red_counter = 0
        
        return self._intervention_history[t],kwargs

class MultiTierPolicy_ACS():
    '''
        A multi-tier policy allows for multiple tiers of lock-downs.
        Attrs:
            tiers (list of dict): a list of the tiers characterized by a dictionary
                with the following entries:
                    {
                        "transmission_reduction": float [0,1)
                        "cocooning": float [0,1)
                        "school_closure": int {0,1}
                    }
            
            lockdown_thresholds (list of list): a list with the thresholds for every
                tier. The list must have n-1 elements if there are n tiers. Each threshold
                is a list of values for evert time step of simulation.
            threshold_type: functional form of the threshold (options are in THRESHOLD_TYPES)
    '''
    def __init__(self, instance, tiers, lockdown_thresholds, acs_thrs, acs_length, acs_lead_time, acs_Q):
        assert len(tiers) == len(lockdown_thresholds)
        self.tiers = tiers
        self.lockdown_thresholds = lockdown_thresholds
        self.lockdown_thresholds_ub = [lockdown_thresholds[i] for i in range(1, len(lockdown_thresholds))]
        self.lockdown_thresholds_ub.append([np.inf] * len(lockdown_thresholds[0]))
        
        self._n = len(self.tiers)
        self._tier_history = None
        self._intervention_history = None
        self._instance = instance
        
        self.red_counter = 0
        
        # ACS parammeters
        self.acs_thrs = acs_thrs
        self.acs_length = acs_length
        self.acs_lead_time = acs_lead_time
        self.acs_Q = acs_Q
    
    @classmethod
    def constant_policy(cls, instance, tiers, constant_thresholds, acs_thrs, acs_length, acs_lead_time, acs_Q):
        T = instance.T
        lockdown_thresholds = [[ct] * T for ct in constant_thresholds]
        return cls(instance, tiers, lockdown_thresholds, acs_thrs, acs_length, acs_lead_time, acs_Q)
    
    def deep_copy(self):
        p = MultiTierPolicy_ACS(self._instance, self.tiers, self.lockdown_thresholds, self.acs_thrs, self.acs_length, self.acs_lead_time, self.acs_Q)
        p.set_tier_history(self._tier_history_copy)
        p.set_intervention_history(self._intervention_history_copy)
        return p
    
    def set_tier_history(self, history):
        # Set history and saves a copy to reset
        self._tier_history = history.copy()
        self._tier_history_copy = history.copy()
    
    def set_intervention_history(self, history):
        # Set history and saves a copy to reset
        self._intervention_history = history.copy()
        self._intervention_history_copy = history.copy()
    
    def reset_history(self):
        # reset history so that a new simulation can be excecuted
        self.set_tier_history(self._tier_history_copy)
        self.set_intervention_history(self._intervention_history_copy)
    
    def compute_cost(self):
        return sum(self.tiers[i]['daily_cost'] for i in self._tier_history if i is not None and i in range(self._n))
    
    def get_tier_history(self):
        return self._tier_history
    
    def get_cap_history(self):
        return self._capacity
    
    def get_interventions_history(self):
        return self._intervention_history
    
    def __repr__(self):
        p_str = str([(self.tiers[i]['name'], self.lockdown_thresholds[i][0], self.lockdown_thresholds[i][-1])
                     for i in range(len(self.tiers))])
        p_str = p_str.replace(' ', '')
        p_str = p_str.replace(',', '_')
        p_str = p_str.replace("'", "")
        p_str = p_str.replace('[', '')
        p_str = p_str.replace('(', '')
        p_str = p_str.replace(']', '')
        p_str = p_str.replace(')', '')
        return p_str
    
    def __call__(self, t, criStat, IH, *args, **kwargs):
        '''
            Function that makes an instance of a policy a callable.
            Args:
                t (int): time period in the simulation
                z (object): deprecated, but maintained to avoid changes in the simulate function
                criStat (ndarray): the trigger statistics, previously daily admission, passed by the simulator
                IH (ndarray): hospitalizations admissions, passed by the simulator
                ** kwargs: additional parameters that are passed and are used elsewhere
        '''
        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - config['moving_avg_len'])
        if len(kwargs["acs_criStat"]) > 0:
            acs_criStat_avg = kwargs["acs_criStat"].sum((1,2))[moving_avg_start:].mean()
        else:
            acs_criStat_avg = 0
        
        # check hospitalization trigger
        if (not kwargs["acs_triggered"]) and (acs_criStat_avg > self.acs_thrs):
            kwargs["acs_triggered"] = True
            for tCap in range(t + self.acs_lead_time,t + self.acs_lead_time + self.acs_length):
                if tCap < len(kwargs["_capacity"]):
                    kwargs["_capacity"][tCap] = kwargs["_capacity"][tCap] + self.acs_Q
        
        if self._tier_history[t] is not None:
            return self._intervention_history[t],kwargs
                
        # enforce the tiers out
        criStat_total = criStat.sum((1, 2))
        criStat_avg = criStat_total[moving_avg_start:].mean()

        effective_threshold = {}
        effective_threshold_ub = {}
        for itier in range(len(self.tiers)):
            effective_threshold[itier] = self.lockdown_thresholds[itier][0]
            effective_threshold_ub[itier] = self.lockdown_thresholds_ub[itier][0]
        if kwargs['fo_tiers'] is None:
            effective_tiers = range(len(self.tiers))
        else:
            if not(kwargs['changed_tiers']):
                effective_tiers = [i for i in range(len(self.tiers)) if i not in kwargs['fo_tiers']]
            else:
                effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers']]
        for tier_ind in range(len(effective_tiers)):
            tier_ix = effective_tiers[tier_ind]
            if tier_ind != len(effective_tiers) - 1:
                effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
            else:
                effective_threshold_ub[tier_ix] = np.inf
        
        current_tier = self._tier_history[t - 1]
        valid_interventions_t = kwargs['feasible_interventions'][t]
        T = self._instance.T
        
        ## NEW
        new_tier = effective_tiers[[
            effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
            for tier_ix in effective_tiers
        ].index(True)]
        
        if new_tier == current_tier:
            t_end = np.minimum(t + 1, T)
            # if it is the first time turning red
            if new_tier == 4:
                # forced out of the red tier
                if self.red_counter >= kwargs["redLimit"]:
                    # if it is the first time turning red
                    if (not(kwargs['changed_tiers'])):
                        kwargs['changed_tiers'] = True
                    effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers'] and i != 4]
                    for tier_ind in range(len(effective_tiers)):
                        tier_ix = effective_tiers[tier_ind]
                        if tier_ind != len(effective_tiers) - 1:
                            effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
                        else:
                            effective_threshold_ub[tier_ix] = np.inf
                    new_tier = effective_tiers[[
                        effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
                        for tier_ix in effective_tiers
                    ].index(True)]
                    t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
        else:
            if new_tier < current_tier:
                IH_total = IH[-1].sum()
                if current_tier != 4:
                    assert_safety_trigger = IH_total < self._instance.hosp_beds * config['safety_threshold_frac']
                    new_tier = new_tier if assert_safety_trigger else current_tier
                    t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
                else:
                    # if it is the first time turning red
                    if (not(kwargs['changed_tiers'])):
                        kwargs['changed_tiers'] = True
                        effective_tiers = [i for i in range(len(self.tiers)) if i in kwargs['after_tiers']]
                        for tier_ind in range(len(effective_tiers)):
                            tier_ix = effective_tiers[tier_ind]
                            if tier_ind != len(effective_tiers) - 1:
                                effective_threshold_ub[tier_ix] = effective_threshold[effective_tiers[tier_ind + 1]]
                            else:
                                effective_threshold_ub[tier_ix] = np.inf
                        new_tier = effective_tiers[[
                            effective_threshold[tier_ix] <= criStat_avg < effective_threshold_ub[tier_ix]
                            for tier_ix in effective_tiers
                        ].index(True)]
                    t_delta = self.tiers[new_tier]['min_enforcing_time']
                t_end = np.minimum(t + t_delta, T)
            else:
                t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
        ## OLD
        # threshold_lb = self.lockdown_thresholds[current_tier][t] if current_tier > 0 else 0
        # threshold_ub = self.lockdown_thresholds[current_tier + 1][t] if current_tier + 1 < self._n - 1 else np.inf
        # if IYIH_avg > threshold_ub:  # bump to the next tier
        #     new_tier = np.minimum(current_tier + 1, self._n - 1)
        #     t_end = np.minimum(t + self.tiers[new_tier]['min_enforcing_time'], T)
        # elif IYIH_avg < threshold_lb:  # relax one tier, if safety trigger allows
        #     IH_total = IH[-1].sum()
        #     assert_safety_trigger = IH_total < self._instance.hosp_beds * config['safety_threshold_frac']
        #     new_tier = np.maximum(current_tier - 1, 0) if assert_safety_trigger else current_tier
        #     t_delta = self.tiers[new_tier]['min_enforcing_time'] if assert_safety_trigger else 1
        #     t_end = np.minimum(t + t_delta, T)
        # else:  # stay in same tier for one more time period
        #     new_tier = current_tier
        #     t_end = np.minimum(t + 1, T)
        self._intervention_history[t:t_end] = valid_interventions_t[new_tier]
        self._tier_history[t:t_end] = new_tier
        if new_tier == 4:
            self.red_counter += (t_end - t)
        else:
            self.red_counter = 0
        
        return self._intervention_history[t],kwargs
