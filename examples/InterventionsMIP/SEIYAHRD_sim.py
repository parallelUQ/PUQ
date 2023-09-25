'''
This module runs a simulation of the SEIYARDH model for a single city,
considering different age groups and seven compartments. This model is
based of Zhanwei's Du model, Meyer's Group.

This module also contains functions to run the simulations in parallel
and a class to properly define a calendar (SimCalendar).
'''
import datetime as dt
import numpy as np
from utils import timeit, roundup
from policies import MultiTierPolicy, MultiTierPolicy_ACS
from InterventionsMIP import config
import copy


def simulate(instance, policy, interventions, seed=-1, **kwargs):
    '''
    Simulates an SIR-type model with seven compartments, multiple age groups,
    and risk different age groups:
    Compartments
        S: Susceptible
        E: Exposed
        IY: Infected symptomatic
        IA: Infected asymptomatic
        IH: Infected hospitalized
        R: Recovered
        D: Death
    Connections between compartments:
        S-E, E-IY, E-IA, IY-IH, IY-R, IA-R, IH-R, IH-D

    Args:
        epi (EpiParams): instance of the parameterization
        T(int): number of time periods
        A(int): number of age groups
        L(int): number of risk groups
        F(int): frequency of the  interventions
        interventions (list of Intervention): list of inteventions
        N(ndarray): total population on each age group
        I0 (ndarray): initial infected people on each age group
        z (ndarray): interventions for each day
        policy (func): callabe to get intervention at time t
        calendar (SimCalendar): calendar of the simulation
        seed (int): seed for random generation. Defaults is -1 which runs
            the simulation in a deterministic fashion.
        kwargs (dict): additional parameters that are passed to the policy function
    '''
    # Local variables
    epi = instance.epi
    
    
    T, A, L = instance.T, instance.A, instance.L
    N, I0 = instance.N, instance.I0
    calendar = instance.cal
    
    # Random stream for stochastic simulations
    if config["det_param"]:
        rnd_epi = None
    else:
        rnd_epi = np.random.RandomState(seed) if seed >= 0 else None
    epi_orig = copy.deepcopy(epi)
    epi_rand = copy.deepcopy(epi)
    epi_rand.update_rnd_stream(rnd_epi)
    epi_orig.update_rnd_stream(None)
    
    # Compartments
    if config['det_history']:
        types = 'float'
    else:
        types = 'int' if seed >= 0 else 'float'
    #types = 'float'
    S = np.zeros((T, A, L), dtype=types)
    E = np.zeros((T, A, L), dtype=types)
    IA = np.zeros((T, A, L), dtype=types)
    IY = np.zeros((T, A, L), dtype=types)
    PA = np.zeros((T, A, L), dtype=types)
    PY = np.zeros((T, A, L), dtype=types)
    IH = np.zeros((T, A, L), dtype=types)
    R = np.zeros((T, A, L), dtype=types)
    D = np.zeros((T, A, L), dtype=types)
    
    # Additional tracking variables (for triggers)
    IYIH = np.zeros((T - 1, A, L))
    
    # Initial Conditions (assumed)
    PY[0] = I0
    #IY[0] = I0
    R[0] = 0
    S[0] = N - PY[0] - IY[0]
    
    # Rates of change
    step_size = config['step_size']
    approx_method = config['approx_method']
    kwargs["acs_triggered"] = False
    kwargs["_capacity"] = [instance.hosp_beds] * instance.T
    
    for t in range(T - 1):
        # Get dynamic intervention and corresponding contact matrix
        k_t, kwargs = policy(t, criStat=eval(kwargs["policy_field"])[:t], IH=IH[:t], **kwargs)
        phi_t = interventions[k_t].phi(calendar.get_day_type(t))
        # if the current time is within the history
        if config['det_history'] and t < len(instance.real_hosp):
            rnd_stream = None
            epi = epi_orig
        else:
            rnd_stream = np.random.RandomState(seed) if (seed >= 0) else None
            epi = epi_rand
        
        if approx_method == 1:
            # directly dividing step_size
            rate_E = epi.sigma_E / step_size
            rate_IYR = np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY for l in range(L)] for a in range(A)]) / step_size
            rate_IAR = np.tile(epi.gamma_IA, (L, A)).transpose() / step_size
            rate_PAIA = np.tile(epi.rho_A, (L, A)).transpose() / step_size
            rate_PYIY = np.tile(epi.rho_Y, (L, A)).transpose() / step_size
            rate_IYH = np.array([[(epi.pi[a, l]) * epi.Eta[a] for l in range(L)] for a in range(A)]) / step_size
            rate_IHD = np.array([[epi.nu[a, l] * epi.mu for l in range(L)] for a in range(A)]) / step_size
            rate_IHR = np.array([[(1 - epi.nu[a, l]) * epi.gamma_IH for l in range(L)] for a in range(A)]) / step_size
        elif approx_method == 2:
            rate_E = discrete_approx(epi.sigma_E, step_size)
            rate_IYR = discrete_approx(
                np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY for l in range(L)] for a in range(A)]), step_size)
            rate_IAR = discrete_approx(np.tile(epi.gamma_IA, (L, A)).transpose(), step_size)
            rate_PAIA = discrete_approx(np.tile(epi.rho_A, (L, A)).transpose(), step_size)
            rate_PYIY = discrete_approx(np.tile(epi.rho_Y, (L, A)).transpose(), step_size)
            rate_IYH = discrete_approx(np.array([[(epi.pi[a, l]) * epi.Eta[a] for l in range(L)] for a in range(A)]),
                                       step_size)
            rate_IHD = discrete_approx(epi.nu*epi.mu,step_size)
            rate_IHR = discrete_approx((1 - epi.nu)*epi.gamma_IH, step_size)
        # Epidemic dynamics
        # Start of daily disctretization in finer time steps
        _S = np.zeros((step_size + 1, A, L), dtype=types)
        _E = np.zeros((step_size + 1, A, L), dtype=types)
        _IA = np.zeros((step_size + 1, A, L), dtype=types)
        _IY = np.zeros((step_size + 1, A, L), dtype=types)
        _PA = np.zeros((step_size + 1, A, L), dtype=types)
        _PY = np.zeros((step_size + 1, A, L), dtype=types)
        _IH = np.zeros((step_size + 1, A, L), dtype=types)
        _R = np.zeros((step_size + 1, A, L), dtype=types)
        _D = np.zeros((step_size + 1, A, L), dtype=types)
        _IYIH = np.zeros((step_size, A, L))
        _S[0] = S[t]
        _E[0] = E[t]
        _IA[0] = IA[t]
        _IY[0] = IY[t]
        _PA[0] = PA[t]
        _PY[0] = PY[t]
        _IH[0] = IH[t]
        _R[0] = R[t]
        _D[0] = D[t]
        
        for _t in range(step_size):
            # Dynamics for dS
            # Vectorized version for efficiency. For-loop version commented below
            temp1 = np.matmul(np.diag(epi.omega_PY), _PY[_t, :, :]) + \
                    np.matmul(np.diag(epi.omega_PA), _PA[_t, :, :]) + \
                    epi.omega_IA * _IA[_t, :, :] + \
                    epi.omega_IY * _IY[_t, :, :]
            temp2 = np.sum(N, axis=1)[np.newaxis].T
            # temp3 = np.divide(np.multiply(discrete_approx(epi.beta * phi_t, step_size), temp1), temp2)
            temp3 = np.divide(np.multiply(epi.beta * phi_t / step_size, temp1), temp2)
            dSprob = np.sum(temp3, axis=(2, 3))
            
            # ================For-loop version of dS ================
            # dSprob = np.zeros((A, L), dtype='float')
            # for a in range(A):
            #     for l in range(L):
            #         beta_t_a = {(a_, l_): epi.beta * phi_t[a, a_, l, l_] / step_size for a_ in range(A) for l_ in range(L)}
            # dSprob[a, l] = sum(beta_t_a[a_, l_]
            #                     * (epi.omega_PY[a_] * _PY[_t, a_, l_] + epi.omega_PA[a_] * _PA[_t, a_, l_]
            #                       + epi.omega_IA * _IA[_t, a_, l_] + epi.omega_IY * _IY[_t, a_, l_])
            #                     / N[a_, :].sum() for a_ in range(A) for l_ in range(L))
            # ================ End for-loop version ================
            
            _dS = rv_gen(rnd_stream, _S[_t], dSprob)
            _S[_t + 1] = _S[_t] - _dS
            
            # Dynamics for E
            E_out = rv_gen(rnd_stream, _E[_t], rate_E)
            _E[_t + 1] = _E[_t] + _dS - E_out
            
            # Dynamics for PA
            EPA = rv_gen(rnd_stream, E_out, (1 - epi.tau))
            PAIA = rv_gen(rnd_stream, _PA[_t], rate_PAIA)
            _PA[_t + 1] = _PA[_t] + EPA - PAIA
            
            # Dynamics for IA
            IAR = rv_gen(rnd_stream, _IA[_t], rate_IAR)
            _IA[_t + 1] = _IA[_t] + PAIA - IAR
            
            # Dynamics for PY
            EPY = E_out - EPA
            PYIY = rv_gen(rnd_stream, _PY[_t], rate_PYIY)
            _PY[_t + 1] = _PY[_t] + EPY - PYIY
            
            # Dynamics for IY
            IYR = rv_gen(rnd_stream, _IY[_t], rate_IYR)
            _IYIH[_t] = rv_gen(rnd_stream, _IY[_t] - IYR, rate_IYH)
            _IY[_t + 1] = _IY[_t] + PYIY - IYR - _IYIH[_t]
            
            # Dynamics for IH
            IHR = rv_gen(rnd_stream, _IH[_t], rate_IHR)
            IHD = rv_gen(rnd_stream, _IH[_t] - IHR, rate_IHD)
            _IH[_t + 1] = _IH[_t] + _IYIH[_t] - IHR - IHD
            
            # Dynamics for R
            _R[_t + 1] = _R[_t] + IHR + IYR + IAR
            
            # Dynamics for D
            _D[_t + 1] = _D[_t] + IHD
        
        # End of the daily disctretization
        S[t + 1] = _S[step_size].copy()
        E[t + 1] = _E[step_size].copy()
        IA[t + 1] = _IA[step_size].copy()
        IY[t + 1] = _IY[step_size].copy()
        PA[t + 1] = _PA[step_size].copy()
        PY[t + 1] = _PY[step_size].copy()
        IH[t + 1] = _IH[step_size].copy()
        R[t + 1] = _R[step_size].copy()
        D[t + 1] = _D[step_size].copy()
        IYIH[t] = _IYIH.sum(axis=0)
        
        # Validate simulation: checks we are not missing people
        # for a in range(A):
        #     for l in range(L):
        #         pop_dif = (
        #             np.sum(S[t, a, l] + E[t, a, l] + IA[t, a, l] + IY[t, a, l] + IH[t, a, l] + R[t, a, l] + D[t, a, l])
        #             - N[a, l])
        #         assert pop_dif < 1E2, f'Pop unbalanced {a} {l} {pop_dif}'
        total_imbalance = np.sum(S[t] + E[t] + IA[t] + IY[t] + IH[t] + R[t] + D[t] + PA[t] + PY[t]) - np.sum(N)
        assert np.abs(total_imbalance) < 1E2, f'fPop unbalanced {total_imbalance}'
    
    # Additional output
    # Change in compartment S, flow from S to E
    dS = S[1:, :] - S[:-1, :]
    # flow from IY to IH
    output = {
        'S': S,
        'E': E,
        'PA': PA,
        'PI': PY,
        'IA': IA,
        'IY': IY,
        'IH': IH,
        'R': R,
        'D': D,
        'dS': dS,
        'IYIH': IYIH,
        'z': policy.get_interventions_history().copy() if isinstance(policy, MultiTierPolicy) or isinstance(policy, MultiTierPolicy_ACS) else None,
        'tier_history': policy.get_tier_history().copy() if isinstance(policy, MultiTierPolicy) or isinstance(policy, MultiTierPolicy_ACS) else None,
        'seed': seed,
        'acs_triggered': kwargs["acs_triggered"],
        'capacity': kwargs["_capacity"]
    }
    
    return output

def simulate_ICU(instance, policy, interventions, seed=-1, **kwargs):
    '''
    Simulates an SIR-type model with seven compartments, multiple age groups,
    and risk different age groups:
    Compartments
        S: Susceptible
        E: Exposed
        IY: Infected symptomatic
        IA: Infected asymptomatic
        IH: Infected hospitalized
        ICU: Infected ICU
        R: Recovered
        D: Death
    Connections between compartments:
        S-E, E-IY, E-IA, IY-IH, IY-R, IA-R, IH-R, IH-ICU, ICU-D, ICU-R

    Args:
        epi (EpiParams): instance of the parameterization
        T(int): number of time periods
        A(int): number of age groups
        L(int): number of risk groups
        F(int): frequency of the  interventions
        interventions (list of Intervention): list of inteventions
        N(ndarray): total population on each age group
        I0 (ndarray): initial infected people on each age group
        z (ndarray): interventions for each day
        policy (func): callabe to get intervention at time t
        calendar (SimCalendar): calendar of the simulation
        seed (int): seed for random generation. Defaults is -1 which runs
            the simulation in a deterministic fashion.
        kwargs (dict): additional parameters that are passed to the policy function
    '''
    # Local variables
    epi = instance.epi
    T, A, L = instance.T, instance.A, instance.L
    N, I0 = instance.N, instance.I0
    calendar = instance.cal
    
    # Random stream for stochastic simulations
    if config["det_param"]:
        rnd_epi = None
    else:
        rnd_epi = np.random.RandomState(seed) if seed >= 0 else None
    epi_orig = copy.deepcopy(epi)
    epi_rand = copy.deepcopy(epi)
    epi_rand.update_rnd_stream(rnd_epi)
    epi_orig.update_rnd_stream(None)
    
    epi_rand.update_hosp_duration()
    epi_orig.update_hosp_duration()
    
    print(instance.param)
    
    # Compartments
    if config['det_history']:
        types = 'float'
    else:
        types = 'int' if seed >= 0 else 'float'
    #types = 'float'
    S = np.zeros((T, A, L), dtype=types)
    E = np.zeros((T, A, L), dtype=types)
    IA = np.zeros((T, A, L), dtype=types)
    IY = np.zeros((T, A, L), dtype=types)
    ITot = np.zeros((T, A, L), dtype=types)
    PA = np.zeros((T, A, L), dtype=types)
    PY = np.zeros((T, A, L), dtype=types)
    IH = np.zeros((T, A, L), dtype=types)
    ICU = np.zeros((T, A, L), dtype=types)
    IHT = np.zeros((T, A, L), dtype=types)
    R = np.zeros((T, A, L), dtype=types)
    D = np.zeros((T, A, L), dtype=types)
    
    # Additional tracking variables (for triggers)
    IYIH = np.zeros((T - 1, A, L))
    IYICU = np.zeros((T - 1, A, L))
    IHICU = np.zeros((T - 1, A, L))
    ToICU = np.zeros((T - 1, A, L))
    ToIHT = np.zeros((T - 1, A, L))
    
    # Initial Conditions (assumed)
    PY[0] = I0
    #IY[0] = I0
    R[0] = 0
    S[0] = N - PY[0] - IY[0]
    
    # Rates of change
    step_size = config['step_size']
    approx_method = config['approx_method']
    kwargs["acs_triggered"] = False
    kwargs["_capacity"] = [instance.hosp_beds] * instance.T
    
    for t in range(T - 1):
        kwargs["acs_criStat"] = eval(kwargs["acs_policy_field"])[:t]
        # Get dynamic intervention and corresponding contact matrix
        k_t, kwargs = policy(t, criStat=eval(kwargs["policy_field"])[:t], IH=IH[:t], **kwargs)
        phi_t = interventions[k_t].phi(calendar.get_day_type(t))
        # if the current time is within the history
        if config['det_history'] and t < len(instance.real_hosp):
            rnd_stream = None
            epi = epi_orig

        else:
            rnd_stream = np.random.RandomState(seed) if (seed >= 0) else None
            epi = epi_rand
        
        if instance.otherInfo == {}:
            if t > kwargs["rd_start"] and t <= kwargs["rd_end"]:
                epi.update_icu_params(kwargs["rd_rate"])
        else:
            epi.update_icu_all(t,instance.otherInfo)
        

        epi.sigma_E = instance.param[0]
        epi.omega_IA = instance.param[1]
        epi.gamma_IY = instance.param[2]
        epi.gamma_IA = instance.param[3]
        
        if approx_method == 1:
            # directly dividing step_size
            rate_E = epi.sigma_E / step_size
            rate_IYR = np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY for l in range(L)] for a in range(A)]) / step_size
            rate_IAR = np.tile(epi.gamma_IA, (L, A)).transpose() / step_size
            rate_PAIA = np.tile(epi.rho_A, (L, A)).transpose() / step_size
            rate_PYIY = np.tile(epi.rho_Y, (L, A)).transpose() / step_size
            rate_IYH = np.array([[(epi.pi[a, l]) * epi.Eta[a] * epi.rIH for l in range(L)] for a in range(A)]) / step_size
            rate_IYICU = np.array([[(epi.pi[a, l]) * epi.Eta[a] * (1 - epi.rIH) for l in range(L)] for a in range(A)]) / step_size
            rate_IHICU = np.array([[epi.nu[a, l] * epi.mu for l in range(L)] for a in range(A)]) / step_size
            rate_IHR = np.array([[(1 - epi.nu[a, l]) * epi.gamma_IH for l in range(L)] for a in range(A)]) / step_size
            rate_ICUD = np.array([[epi.nu_ICU[a, l] * epi.mu_ICU for l in range(L)] for a in range(A)]) / step_size
            rate_ICUR = np.array([[(1 - epi.nu_ICU[a, l]) * epi.gamma_ICU for l in range(L)] for a in range(A)]) / step_size

        elif approx_method == 2:
            rate_E = discrete_approx(epi.sigma_E, step_size)
            rate_IYR = discrete_approx(
                np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY for l in range(L)] for a in range(A)]), step_size)
            rate_IAR = discrete_approx(np.tile(epi.gamma_IA, (L, A)).transpose(), step_size)
            rate_PAIA = discrete_approx(np.tile(epi.rho_A, (L, A)).transpose(), step_size)
            rate_PYIY = discrete_approx(np.tile(epi.rho_Y, (L, A)).transpose(), step_size)
            rate_IYH = discrete_approx(np.array([[(epi.pi[a, l]) * epi.Eta[a] * epi.rIH for l in range(L)] for a in range(A)]),
                                       step_size)
            rate_IYICU = discrete_approx(np.array([[(epi.pi[a, l]) * epi.Eta[a] * (1 - epi.rIH) for l in range(L)] for a in range(A)]),
                                       step_size)
            rate_IHICU = discrete_approx(epi.nu*epi.mu,step_size)
            rate_IHR = discrete_approx((1 - epi.nu)*epi.gamma_IH, step_size)
            rate_ICUD = discrete_approx(epi.nu_ICU*epi.mu_ICU,step_size)
            rate_ICUR = discrete_approx((1 - epi.nu_ICU)*epi.gamma_ICU, step_size)
        # Epidemic dynamics
        # Start of daily disctretization in finer time steps
        _S = np.zeros((step_size + 1, A, L), dtype=types)
        _E = np.zeros((step_size + 1, A, L), dtype=types)
        _IA = np.zeros((step_size + 1, A, L), dtype=types)
        _IY = np.zeros((step_size + 1, A, L), dtype=types)
        _ITot = np.zeros((step_size + 1, A, L), dtype=types)
        _PA = np.zeros((step_size + 1, A, L), dtype=types)
        _PY = np.zeros((step_size + 1, A, L), dtype=types)
        _IH = np.zeros((step_size + 1, A, L), dtype=types)
        _ICU = np.zeros((step_size + 1, A, L), dtype=types)
        _R = np.zeros((step_size + 1, A, L), dtype=types)
        _D = np.zeros((step_size + 1, A, L), dtype=types)
        _IYIH = np.zeros((step_size, A, L))
        _IYICU = np.zeros((step_size, A, L))
        _IHICU = np.zeros((step_size, A, L))
        _ToICU = np.zeros((step_size, A, L))
        _ToIHT = np.zeros((step_size, A, L))
        _S[0] = S[t]
        _E[0] = E[t]
        _IA[0] = IA[t]
        _IY[0] = IY[t]
        _ITot[0] = ITot[t]
        _PA[0] = PA[t]
        _PY[0] = PY[t]
        _IH[0] = IH[t]
        _ICU[0] = ICU[t]
        _R[0] = R[t]
        _D[0] = D[t]
        
        for _t in range(step_size):
            # Dynamics for dS
            # Vectorized version for efficiency. For-loop version commented below
            temp1 = np.matmul(np.diag(epi.omega_PY), _PY[_t, :, :]) + \
                    np.matmul(np.diag(epi.omega_PA), _PA[_t, :, :]) + \
                    epi.omega_IA * _IA[_t, :, :] + \
                    epi.omega_IY * _IY[_t, :, :]
            temp2 = np.sum(N, axis=1)[np.newaxis].T
            # temp3 = np.divide(np.multiply(discrete_approx(epi.beta * phi_t, step_size), temp1), temp2)
            temp3 = np.divide(np.multiply(epi.beta * phi_t / step_size, temp1), temp2)
            dSprob = np.sum(temp3, axis=(2, 3))
            
            # ================For-loop version of dS ================
            # dSprob = np.zeros((A, L), dtype='float')
            # for a in range(A):
            #     for l in range(L):
            #         beta_t_a = {(a_, l_): epi.beta * phi_t[a, a_, l, l_] / step_size for a_ in range(A) for l_ in range(L)}
            # dSprob[a, l] = sum(beta_t_a[a_, l_]
            #                     * (epi.omega_PY[a_] * _PY[_t, a_, l_] + epi.omega_PA[a_] * _PA[_t, a_, l_]
            #                       + epi.omega_IA * _IA[_t, a_, l_] + epi.omega_IY * _IY[_t, a_, l_])
            #                     / N[a_, :].sum() for a_ in range(A) for l_ in range(L))
            # ================ End for-loop version ================
            
            _dS = rv_gen(rnd_stream, _S[_t], dSprob)
            _S[_t + 1] = _S[_t] - _dS
            
            # Dynamics for E
            E_out = rv_gen(rnd_stream, _E[_t], rate_E)
            _E[_t + 1] = _E[_t] + _dS - E_out
            
            # Dynamics for PA
            EPA = rv_gen(rnd_stream, E_out, (1 - epi.tau))
            PAIA = rv_gen(rnd_stream, _PA[_t], rate_PAIA)
            _PA[_t + 1] = _PA[_t] + EPA - PAIA
            
            # Dynamics for IA
            IAR = rv_gen(rnd_stream, _IA[_t], rate_IAR)
            _IA[_t + 1] = _IA[_t] + PAIA - IAR
            
            # Dynamics for PY
            EPY = E_out - EPA
            PYIY = rv_gen(rnd_stream, _PY[_t], rate_PYIY)
            _PY[_t + 1] = _PY[_t] + EPY - PYIY
            
            # Dynamics for IY
            IYR = rv_gen(rnd_stream, _IY[_t], rate_IYR)
            _IYIH[_t] = rv_gen(rnd_stream, _IY[_t] - IYR, rate_IYH)
            _IYICU[_t] = rv_gen(rnd_stream, _IY[_t] - IYR - _IYIH[_t], rate_IYICU)
            _IY[_t + 1] = _IY[_t] + PYIY - IYR - _IYIH[_t] - _IYICU[_t]
            
            _ITot[_t + 1] = _IY[_t + 1] + _IA[_t + 1]
            
            # Dynamics for IH
            IHR = rv_gen(rnd_stream, _IH[_t], rate_IHR)
            _IHICU[_t] = rv_gen(rnd_stream, _IH[_t] - IHR, rate_IHICU)
            _IH[_t + 1] = _IH[_t] + _IYIH[_t] - IHR - _IHICU[_t]
            
            # Dynamics for ICU
            ICUR = rv_gen(rnd_stream, _ICU[_t], rate_ICUR)
            ICUD = rv_gen(rnd_stream, _ICU[_t] - ICUR, rate_ICUD)
            _ICU[_t + 1] = _ICU[_t] + _IHICU[_t] - ICUD - ICUR + _IYICU[_t]
            _ToICU[_t] = _IYICU[_t] + _IHICU[_t]
            _ToIHT[_t] = _IYICU[_t] + _IYIH[_t]
            
            # Dynamics for R
            _R[_t + 1] = _R[_t] + IHR + IYR + IAR + ICUR
            
            # Dynamics for D
            _D[_t + 1] = _D[_t] + ICUD
        
        # End of the daily disctretization
        S[t + 1] = _S[step_size].copy()
        E[t + 1] = _E[step_size].copy()
        IA[t + 1] = _IA[step_size].copy()
        IY[t + 1] = _IY[step_size].copy()
        ITot[t + 1] = _ITot[step_size].copy()
        PA[t + 1] = _PA[step_size].copy()
        PY[t + 1] = _PY[step_size].copy()
        IH[t + 1] = _IH[step_size].copy()
        ICU[t + 1] = _ICU[step_size].copy()
        IHT[t + 1] = IH[t + 1] + ICU[t + 1]
        R[t + 1] = _R[step_size].copy()
        D[t + 1] = _D[step_size].copy()
        IYIH[t] = _IYIH.sum(axis=0)
        IYICU[t] = _IYICU.sum(axis=0)
        IHICU[t] = _IHICU.sum(axis=0)
        ToICU[t] = _ToICU.sum(axis=0)
        ToIHT[t] = _ToIHT.sum(axis=0)
        
        # Validate simulation: checks we are not missing people
        # for a in range(A):
        #     for l in range(L):
        #         pop_dif = (
        #             np.sum(S[t, a, l] + E[t, a, l] + IA[t, a, l] + IY[t, a, l] + IH[t, a, l] + R[t, a, l] + D[t, a, l])
        #             - N[a, l])
        #         assert pop_dif < 1E2, f'Pop unbalanced {a} {l} {pop_dif}'
        total_imbalance = np.sum(S[t] + E[t] + IA[t] + IY[t] + IH[t] + R[t] + D[t] + PA[t] + PY[t] + ICU[t]) - np.sum(N)
        assert np.abs(total_imbalance) < 1E2, f'fPop unbalanced {total_imbalance}'
    
    # Additional output
    # Change in compartment S, flow from S to E
    dS = S[1:, :] - S[:-1, :]
    # flow from IY to IH
    output = {
        'S': S,
        'E': E,
        'PA': PA,
        'PI': PY,
        'IA': IA,
        'IY': IY,
        'IH': IH,
        'R': R,
        'D': D,
        'ICU': ICU,
        'dS': dS,
        'IYIH': IYIH,
        'IYICU': IYICU,
        'IHICU': IHICU,
        'z': policy.get_interventions_history().copy() if isinstance(policy, MultiTierPolicy) or isinstance(policy, MultiTierPolicy_ACS) else None,
        'tier_history': policy.get_tier_history().copy() if isinstance(policy, MultiTierPolicy) or isinstance(policy, MultiTierPolicy_ACS) else None,
        'seed': seed,
        'acs_triggered': kwargs["acs_triggered"],
        'capacity': kwargs["_capacity"],
        'ToICU': ToICU,
        'ToIHT': ToIHT,
        'IHT': IHT
    }
    
    return output



def simulate_active(instance, policy, interventions, seed=-1, **kwargs):
    '''
    Simulates an SIR-type model with seven compartments, multiple age groups,
    and risk different age groups:
    Compartments
        S: Susceptible
        E: Exposed
        IY: Infected symptomatic
        IA: Infected asymptomatic
        IH: Infected hospitalized
        R: Recovered
        D: Death
    Connections between compartments:
        S-E, E-IY, E-IA, IY-IH, IY-R, IA-R, IH-R, IH-D

    Args:
        epi (EpiParams): instance of the parameterization
        T(int): number of time periods
        A(int): number of age groups
        L(int): number of risk groups
        F(int): frequency of the  interventions
        interventions (list of Intervention): list of inteventions
        N(ndarray): total population on each age group
        I0 (ndarray): initial infected people on each age group
        z (ndarray): interventions for each day
        policy (func): callabe to get intervention at time t
        calendar (SimCalendar): calendar of the simulation
        seed (int): seed for random generation. Defaults is -1 which runs
            the simulation in a deterministic fashion.
        kwargs (dict): additional parameters that are passed to the policy function
    '''
    # Local variables
    epi = instance.epi
    T, A, L = instance.T, instance.A, instance.L
    N, I0 = instance.N, instance.I0
    calendar = instance.cal
    randTest = epi.qInt['randTest']
    testStart = epi.qInt['testStart']
    qRate = epi.qInt['qRate']
    
    # Random stream for stochastic simulations
    if config["det_param"]:
        rnd_epi = None
    else:
        rnd_epi = np.random.RandomState(seed) if seed >= 0 else None
    epi_orig = copy.deepcopy(epi)
    epi_rand = copy.deepcopy(epi)
    epi_rand.update_rnd_stream(rnd_epi)
    epi_orig.update_rnd_stream(None)
    rnd_q = np.random.RandomState()
    
    # Compartments
    if config['det_history']:
        types = 'float'
    else:
        types = 'int' if seed >= 0 else 'float'
    #types = 'float'
    S = np.zeros((T, A, L), dtype=types)
    E = np.zeros((T, A, L), dtype=types)
    IA = np.zeros((T, A, L), dtype=types)
    IY = np.zeros((T, A, L), dtype=types)
    PA = np.zeros((T, A, L), dtype=types)
    PY = np.zeros((T, A, L), dtype=types)
    IH = np.zeros((T, A, L), dtype=types)
    R = np.zeros((T, A, L), dtype=types)
    D = np.zeros((T, A, L), dtype=types)
    
    # Additional tracking variables (for triggers)
    IYIH = np.zeros((T - 1, A, L))
    IHR = np.zeros((T - 1, A, L))
    
    # test positive results record
    tY_rec = np.zeros((T, A, L), dtype = types)
    tA_rec = np.zeros((T, A, L), dtype = types)
    QIY = np.zeros((T, A, L), dtype=types)
    QPY = np.zeros((T, A, L), dtype=types)
    QIA = np.zeros((T, A, L), dtype=types)
    QPA = np.zeros((T, A, L), dtype=types)
    QR = np.zeros((T, A, L), dtype=types)
    tQ = np.zeros(T, dtype=types)  # total number of people in the quarantine sectors
    
    # Initial Conditions (assumed)
    PY[0] = I0
    #IY[0] = I0
    R[0] = 0
    S[0] = N - PY[0] - IY[0]
    
    # Rates of change
    step_size = config['step_size']
    approx_method = config['approx_method']
    active_intervention = config['active_intervention']
    
    for t in range(T - 1):
        kwargs["acs_criStat"] = eval(kwargs["acs_policy_field"])
        # Get dynamic intervention and corresponding contact matrix
        k_t = policy(t, criStat=eval(kwargs["policy_field"])[:t], IH=IH[:t], **kwargs)
        phi_t = interventions[k_t].phi(calendar.get_day_type(t))
        # if the current time is within the history
        if config['det_history'] and t < len(instance.real_hosp):
            rnd_stream = None
            epi = epi_orig
        else:
            rnd_stream = np.random.RandomState(seed) if (seed >= 0) else None
            epi = epi_rand
        
        if approx_method == 1:
            # directly dividing step_size
            rate_E = epi.sigma_E / step_size
            rate_IYR = np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY for l in range(L)] for a in range(A)]) / step_size
            rate_IAR = np.tile(epi.gamma_IA, (L, A)).transpose() / step_size
            rate_PAIA = np.tile(epi.rho_A, (L, A)).transpose() / step_size
            rate_PYIY = np.tile(epi.rho_Y, (L, A)).transpose() / step_size
            rate_IYH = np.array([[(epi.pi[a, l]) * epi.Eta[a] for l in range(L)] for a in range(A)]) / step_size
            rate_IHD = np.array([[epi.nu[a, l] * epi.mu for l in range(L)] for a in range(A)]) / step_size
            rate_IHR = np.array([[(1 - epi.nu[a, l]) * epi.gamma_IH for l in range(L)] for a in range(A)]) / step_size
            rate_QA = np.tile(epi.QA, (L, A)).transpose() / step_size
        elif approx_method == 2:
            rate_E = discrete_approx(epi.sigma_E, step_size)
            rate_IYR = discrete_approx(
                np.array([[(1 - epi.pi[a, l]) * epi.gamma_IY for l in range(L)] for a in range(A)]), step_size)
            rate_IAR = discrete_approx(np.tile(epi.gamma_IA, (L, A)).transpose(), step_size)
            rate_PAIA = discrete_approx(np.tile(epi.rho_A, (L, A)).transpose(), step_size)
            rate_PYIY = discrete_approx(np.tile(epi.rho_Y, (L, A)).transpose(), step_size)
            rate_IYH = discrete_approx(np.array([[(epi.pi[a, l]) * epi.Eta[a] for l in range(L)] for a in range(A)]),
                                       step_size)
            rate_IHD = discrete_approx(np.array([[epi.nu[a, l] * epi.mu for l in range(L)] for a in range(A)]),
                                       step_size)
            rate_IHR = discrete_approx(
                np.array([[(1 - epi.nu[a, l]) * epi.gamma_IH for l in range(L)] for a in range(A)]), step_size)
            rate_QA = discrete_approx(np.tile(epi.QA, (L, A)).transpose(), step_size)
        
        # active intervention: suppose we can cut down transmission qRate of tested positive
        if t >= instance.cal.calendar_ix[testStart]:
            # total population up for test is total population - IH[t] - sum(IHR) - D[t]
            totalTestPop = instance.N.sum() - IH[t].sum() - IHR.sum() - D[t].sum() - tQ[t]

            tIY = rv_gen(rnd_q, randTest, IY[t]/totalTestPop)
            tPY = rv_gen(rnd_q, randTest, PY[t]/totalTestPop)
            tIA = rv_gen(rnd_q, randTest, IA[t]/totalTestPop)
            tPA = rv_gen(rnd_q, randTest, PA[t]/totalTestPop)
            tY_rec[t] = tIY + tPY
            tA_rec[t] = tIA + tPA
            
            # if there is active intervention, qRate default 0
            if active_intervention:
                qIY = rv_gen(rnd_q, tIY, qRate["IY"])
                qPY = rv_gen(rnd_q, tPY, qRate["PY"])
                qIA = rv_gen(rnd_q, tIA, qRate["IA"])
                qPA = rv_gen(rnd_q, tPA, qRate["PA"])
            else:
                qIY = 0.0
                qPY = 0.0
                qIA = 0.0
                qPA = 0.0
        else:
            qIY = 0.0
            qPY = 0.0
            qIA = 0.0
            qPA = 0.0
        
        # Epidemic dynamics
        # Start of daily disctretization in finer time steps
        _S = np.zeros((step_size + 1, A, L), dtype=types)
        _E = np.zeros((step_size + 1, A, L), dtype=types)
        _IA = np.zeros((step_size + 1, A, L), dtype=types)
        _IY = np.zeros((step_size + 1, A, L), dtype=types)
        _PA = np.zeros((step_size + 1, A, L), dtype=types)
        _PY = np.zeros((step_size + 1, A, L), dtype=types)
        _IH = np.zeros((step_size + 1, A, L), dtype=types)
        _R = np.zeros((step_size + 1, A, L), dtype=types)
        _D = np.zeros((step_size + 1, A, L), dtype=types)
        _IYIH = np.zeros((step_size, A, L))
        _IHR = np.zeros((step_size, A, L))
        
        _QIY = np.zeros((step_size + 1, A, L), dtype=types)
        _QPY = np.zeros((step_size + 1, A, L), dtype=types)
        _QIA = np.zeros((step_size + 1, A, L), dtype=types)
        _QPA = np.zeros((step_size + 1, A, L), dtype=types)
        _QR = np.zeros((step_size + 1, A, L), dtype=types)
        _QIYIH = np.zeros((step_size, A, L))
        
        _S[0] = S[t]
        _E[0] = E[t]
        _IA[0] = IA[t] - qIA
        _IY[0] = IY[t] - qIY
        _PA[0] = PA[t] - qPA
        _PY[0] = PY[t] - qPY
        _IH[0] = IH[t]
        _R[0] = R[t]
        _D[0] = D[t]
        _QIY[0] = QIY[t]
        _QIA[0] = QIA[t]
        _QPY[0] = QPY[t]
        _QPA[0] = QPA[t]
        _QR[0] = QR[t]
        
        for _t in range(step_size):
            # Dynamics for QIY, QIA, QPY, QPA
            QPAR = rv_gen(rnd_stream, _QPA[_t], rate_QA)
            QIAR = rv_gen(rnd_stream, _QIA[_t], rate_QA)
            QPYIY = rv_gen(rnd_stream, _QPY[_t], rate_PYIY)
            QIYR = rv_gen(rnd_stream, _QIY[_t], rate_IYR)
            _QIYIH[_t] = rv_gen(rnd_stream, _QIY[_t] - QIYR, rate_IYH)
            
            # Dynamics for dS
            # TODO Vectorized dS might save 50% of the time
            temp1 = np.matmul(np.diag(epi.omega_PY), _PY[_t, :, :]) + \
                    np.matmul(np.diag(epi.omega_PA), _PA[_t, :, :]) + \
                    epi.omega_IA * _IA[_t, :, :] + \
                    epi.omega_IY * _IY[_t, :, :]
            temp2 = np.sum(N, axis=1)[np.newaxis].T
            temp3 = np.divide(np.multiply(epi.beta * phi_t / step_size, temp1), temp2)
            dSprob = np.sum(temp3, axis=(2, 3))
            
            _dS = rv_gen(rnd_stream, _S[_t], dSprob)
            _S[_t + 1] = _S[_t] - _dS
            
            # Dynamics for E
            E_out = rv_gen(rnd_stream, _E[_t], rate_E)
            _E[_t + 1] = _E[_t] + _dS - E_out
            
            # Dynamics for PA
            EPA = rv_gen(rnd_stream, E_out, (1 - epi.tau))
            PAIA = rv_gen(rnd_stream, _PA[_t], rate_PAIA)
            _PA[_t + 1] = _PA[_t] + EPA - PAIA
            
            # Dynamics for IA
            IAR = rv_gen(rnd_stream, _IA[_t], rate_IAR)
            _IA[_t + 1] = _IA[_t] + PAIA - IAR
            
            # Dynamics for PY
            EPY = E_out - EPA
            PYIY = rv_gen(rnd_stream, _PY[_t], rate_PYIY)
            _PY[_t + 1] = _PY[_t] + EPY - PYIY
            
            # Dynamics for IY
            IYR = rv_gen(rnd_stream, _IY[_t], rate_IYR)
            _IYIH[_t] = rv_gen(rnd_stream, _IY[_t] - IYR, rate_IYH)
            _IY[_t + 1] = _IY[_t] + PYIY - IYR - _IYIH[_t]
            
            # Dynamics for IH
            _IHR[_t] = rv_gen(rnd_stream, _IH[_t], rate_IHR)
            IHD = rv_gen(rnd_stream, _IH[_t] - _IHR[_t], rate_IHD)
            _IH[_t + 1] = _IH[_t] + _IYIH[_t] + _QIYIH[_t] - _IHR[_t] - IHD
            
            # Dynamics for R
            _R[_t + 1] = _R[_t] + _IHR[_t] + IYR + IAR
            
            # Dynamics for D
            _D[_t + 1] = _D[_t] + IHD
            
            # Update the quarantine states
            _QPA[_t + 1] = _QPA[_t] - QPAR
            _QIA[_t + 1] = _QIA[_t] - QIAR
            _QPY[_t + 1] = _QPY[_t] - QPYIY
            _QIY[_t + 1] = _QIY[_t] + QPYIY - QIYR - _QIYIH[_t]
            _QR[_t + 1] = _QR[_t] + QPAR + QIAR + QIYR
        
        # End of the daily disctretization
        S[t + 1] = _S[step_size].copy()
        E[t + 1] = _E[step_size].copy()
        IA[t + 1] = _IA[step_size].copy()
        IY[t + 1] = _IY[step_size].copy()
        PA[t + 1] = _PA[step_size].copy()
        PY[t + 1] = _PY[step_size].copy()
        IH[t + 1] = _IH[step_size].copy()
        R[t + 1] = _R[step_size].copy()
        D[t + 1] = _D[step_size].copy()
        IYIH[t] = _IYIH.sum(axis=0) + _QIYIH.sum(axis=0)
        IHR[t] = _IHR.sum(axis=0)
        
        QIA[t + 1] = _QIA[step_size].copy() + qIA
        QIY[t + 1] = _QIY[step_size].copy() + qIY
        QPA[t + 1] = _QPA[step_size].copy() + qPA
        QPY[t + 1] = _QPY[step_size].copy() + qPY
        QR[t + 1] = _QR[step_size].copy()
        tQ[t + 1] = QIA[t + 1].sum() + QIY[t + 1].sum() + QPA[t + 1].sum() + QPY[t + 1].sum() + QR[t + 1].sum()
        
        # Validate simulation: checks we are not missing people
        # for a in range(A):
        #     for l in range(L):
        #         pop_dif = (
        #             np.sum(S[t, a, l] + E[t, a, l] + IA[t, a, l] + IY[t, a, l] + IH[t, a, l] + R[t, a, l] + D[t, a, l])
        #             - N[a, l])
        #         assert pop_dif < 1E2, f'Pop unbalanced {a} {l} {pop_dif}'
        # total_imbalance = np.sum(S[t] + E[t] + IA[t] + IY[t] + IH[t] + R[t] + D[t] + PA[t] + PY[t]) - np.sum(N)
        # assert np.abs(total_imbalance) < 1E2, f'fPop unbalanced {total_imbalance}'
    
    # Additional output
    # Change in compartment S, flow from S to E
    dS = S[1:, :] - S[:-1, :]
    # flow from IY to IH
    output = {
        'S': S,
        'E': E,
        'PA': PA,
        'PY': PY,
        'IA': IA,
        'IY': IY,
        'IH': IH,
        'R': R,
        'tA': tA_rec,
        'tY': tY_rec,
        'QPA': QPA,
        'QPY': QPY,
        'QIA': QIA,
        'QIY': QIY,
        'IH': IH,
        'R': R,
        'D': D,
        'dS': dS,
        'IYIH': IYIH,
        'z': policy.get_interventions_history().copy() if isinstance(policy, MultiTierPolicy) or isinstance(policy, MultiTierPolicy_ACS) else None,
        'tier_history': policy.get_tier_history().copy() if isinstance(policy, MultiTierPolicy) or isinstance(policy, MultiTierPolicy_ACS) else None,
        'seed': seed
    }
    
    return output


def rv_gen(rnd_stream, n, p, round_opt=1):
    if rnd_stream is None:
        return n * p
    else:
        if round_opt:
            nInt = np.round(n)
            return rnd_stream.binomial(nInt.astype(int), p)
        else:
            return rnd_stream.binomial(n, p)


def system_simulation(mp_sim_input):
    '''
        Simulation function that gets mapped when running simulations in parallel.
        Args:
            mp_sim_input (tuple):
                instance, policy, cost_func, interventions, kwargs (as a dict)
        Returns:
            out_sim (dict): output of the simulation
            policy_cost (float): evaluation of cost_func
            policy (object): the policy used in the simulation
            seed (int): seed used in the simulation
            kwargs (dict): additional parameters used
    '''
    instance, policy, cost_func, interventions, seed, kwargs = mp_sim_input
    if kwargs['sim_method'] == 1:
        if kwargs['icu_trigger']:
            out_sim = simulate_ICU(instance, policy, interventions, seed, **kwargs)
        else:
            out_sim = simulate(instance, policy, interventions, seed, **kwargs)
    elif kwargs['sim_method'] == 2:
        out_sim = simulate_active(instance, policy, interventions, seed, **kwargs)
    policy_cost, cost_info = cost_func(instance, policy, out_sim, **kwargs)
    kwargs_new = kwargs.copy()
    kwargs_new["cost_info"] = cost_info
    return out_sim, policy_cost, policy, seed, kwargs_new


@timeit
def simulate_p(mp_pool, input_iter):
    '''
    Launches simulation in parallel
    Args:
        mp_pool (Pool): pool to parallelize
        input_ite (iterator): iterator with the inputs to parallelize.
            Input signature: 
                instance, policy, cost_func, interventions, kwargs (as a dict)
    Return:
        list of outputs is a tuple with:
            out_sim (dict): output of the simulation
            policy_cost (float): evaluation of cost_func
            policy (object): the policy used in the simulation
            seed (int): seed used in the simulation
            kwargs (dict): additional parameters used

    '''
    if mp_pool is None:
        results = []
        for sim_input in input_iter:
            results.append(system_simulation(sim_input))
        return results
    else:
        results = mp_pool.map(system_simulation, input_iter)
        return results


def dummy_cost(*args, **kwargs):
    return 0


def fix_policy(t, z, *args, **kwargs):
    '''
        Returns the intervention according to a
        fix policy z
        Args:
            t (int): time of the intervention
            z (ndarray): fix policy
    '''
    return z[t]


def hosp_based_policy(t, z, opt_phase, moving_avg_len, IYIH_threshold, hosp_level_release, baseline_enforcing_time,
                      lockdown_enforcing_time, feasible_interventions, SD_state, IYIH, IH, **kwargs):
    '''
        Lock-down and relaxation policy function. This function returns the
        intervention to be used at time t, according to the thresholds that
        are given as paramters.
        Args:
            t (int): time step of the simulation
            z (ndarray): vector with the interventions
            opt_phase (bool): True if optimization phase is happening, false otherwise
            moving_avg_len (int): number of days to compute IYIH moving average
            IYIH_threshold (list): threshold values for each time period
            hosp_level_release (float): value of the safety trigger for total hospitalizations
            baseline_enforcing_time (int): number of days relaxation is enforced
            lockdown_enforcing_time (int): number of days lockdown is enforced
            feasible_interventions (list of dict): list of feasible interventions. Dictionary
                has the signature of {'H': int, 'L': int}.
            SD_state (list): toggle history to keep track of whether at time t there is lockdown
                or relaxation.
            IYIH (ndarray): daily admissions, passed by the simulator
            IH (ndarray): hospitalizations admissions, passed by the simulator
            ** kwargs: additional parameters that are passed and are used elsewhere
    '''
    # If the state is already set, apply it right away
    if SD_state[t] is not None or t == 0:
        return z[t]
    
    # Compute daily admissions moving average
    moving_avg_start = np.maximum(0, t - moving_avg_len)
    IYIH_total = IYIH.sum((1, 2))
    IYIH_avg = IYIH_total[moving_avg_start:].mean()
    
    # Get threshold for time t
    hosp_rate_threshold = IYIH_threshold[t]
    # Get valid intervention at time t
    valid_t = feasible_interventions[t]
    
    if SD_state[t - 1] == 'L':  # If relaxation is currently in place
        if IYIH_avg >= hosp_rate_threshold:
            t_end = np.minimum(t + lockdown_enforcing_time, len(z))
            z[t:t_end] = valid_t['H']  # Turn on heavy SD from t to t_end
            SD_state[t:t_end] = 'H'
        else:
            z[t] = valid_t['L']  # Keep baseline
            SD_state[t] = 'L'
    elif SD_state[t - 1] == 'H':  # If lock-down is currently in place
        IH_total = IH[-1].sum()
        if IH_total <= hosp_level_release and IYIH_avg < hosp_rate_threshold:
            t_end = np.minimum(t + baseline_enforcing_time, len(z))
            z[t:t_end] = valid_t['L']  # Turn off heavy SD from t to t_end
            SD_state[t:t_end] = 'L'
        else:
            z[t] = valid_t['H']  # keep heavy social distance
            SD_state[t] = 'H'
    else:
        raise f'Unknown state/intervention {SD_state[t-1]} {z[t-1]}'
    
    return z[t]


WEEKDAY = 1
WEEKEND = 2
HOLIDAY = 3
LONG_HOLIDAY = 4


class SimCalendar():
    '''
        A simulation calendar to map time steps to days. This class helps
        to determine whether a time step t is a weekday or a weekend, as well
        as school calendars.

        Attrs:
        start (datetime): start date of the simulation
        calendar (list): list of datetime for every time step
    '''
    def __init__(self, start_date, sim_length):
        '''
            Arg
        '''
        self.start = start_date
        self.calendar = [self.start + dt.timedelta(days=t) for t in range(sim_length)]
        self.calendar_ix = {d: d_ix for (d_ix, d) in enumerate(self.calendar)}
        self._is_weekday = [d.weekday() not in [5, 6] for d in self.calendar]
        self._day_type = [WEEKDAY if iw else WEEKEND for iw in self._is_weekday]
        self.lockdown = None
        self.schools_closed = None
        self.fixed_transmission_reduction = None
        self.fixed_cocooning = None
        self.month_starts = self.get_month_starts()
    
    def is_weekday(self, t):
        '''
            True if t is a weekday, False otherwise
        '''
        return self._is_weekday[t]
    
    def get_day_type(self, t):
        '''
            Returns the date type with the convention of the class
        '''
        return self._day_type[t]
    
    def load_predefined_lockdown(self, lockdown_blocks):
        '''
            Loads fixed decisions on predefined lock-downs and saves
            it on attribute lockdown.
            Args:
                lockdown_blocks (list of tuples): a list with blocks in which predefined lockdown is enacted
                (e.g. [(datetime.date(2020,3,24),datetime.date(2020,8,28))])
            
        '''
        self.lockdown = []
        for d in self.calendar:
            closedDay = False
            for blNo in range(len(lockdown_blocks)):
                if d >= lockdown_blocks[blNo][0] and d <= lockdown_blocks[blNo][1]:
                    closedDay = True
            self.lockdown.append(closedDay)
    
    def load_school_closure(self, school_closure_blocks):
        '''
            Load fixed decisions on school closures and saves
            it on attribute schools_closed
            Args:
                school_closure_blocks (list of tuples): a list with blocks in which schools are closed
                (e.g. [(datetime.date(2020,3,24),datetime.date(2020,8,28))])
        '''
        self.schools_closed = []
        for d in self.calendar:
            closedDay = False
            for blNo in range(len(school_closure_blocks)):
                if d >= school_closure_blocks[blNo][0] and d <= school_closure_blocks[blNo][1]:
                    closedDay = True
            self.schools_closed.append(closedDay)
    
    def load_fixed_transmission_reduction(self, ts_transmission_reduction, present_date=dt.datetime.today()):
        '''
            Load fixed decisions on transmission reduction and saves it on attribute fixed_transmission_reduction.
            If a value is not given, the transmission reduction is None.
            Args:
                ts_transmission_reduction (list of tuple): a list with the time series of
                    transmission reduction (datetime, float).
                present_date (datetime): reference date so that all dates before must have a
                    transmission reduction defined
        '''
        self.fixed_transmission_reduction = [None for d in self.calendar]
        for (d, tr) in ts_transmission_reduction:
            if d in self.calendar_ix:
                d_ix = self.calendar_ix[d]
                self.fixed_transmission_reduction[d_ix] = tr
                
    def load_fixed_cocooning(self, ts_cocooning, present_date=dt.datetime.today()):
        '''
            Load fixed decisions on transmission reduction and saves it on attribute fixed_transmission_reduction.
            If a value is not given, the transmission reduction is None.
            Args:
                ts_cocooning (list of tuple): a list with the time series of
                    transmission reduction (datetime, float).
                present_date (datetime): reference date so that all dates before must have a
                    transmission reduction defined
        '''
        self.fixed_cocooning = [None for d in self.calendar]
        for (d, tr) in ts_cocooning:
            if d in self.calendar_ix:
                d_ix = self.calendar_ix[d]
                self.fixed_cocooning[d_ix] = tr

    
    def load_holidays(self, holidays=[], long_holidays=[]):
        '''
            Change the day_type for holidays
        '''
        for hd in holidays:
            dt_hd = dt.datetime(hd.year, hd.month, hd.day)
            if dt_hd in self.calendar:
                self._day_type[self.calendar_ix[dt_hd]] = HOLIDAY
        
        for hd in long_holidays:
            dt_hd = dt.datetime(hd.year, hd.month, hd.day)
            if dt_hd in self.calendar:
                self._day_type[self.calendar_ix[dt_hd]] = LONG_HOLIDAY
    
    def get_month_starts(self):
        '''
            Get a list of first days of months
        '''
        month_starts = []
        
        currentTemp = get_next_month(self.start)
        while currentTemp <= self.calendar[-1]:
            month_starts.append(self.calendar_ix[currentTemp])
            currentTemp = get_next_month(currentTemp)
        
        return month_starts
    
    def __len__(self):
        return len(self.calendar)


def get_next_month(dateG):
    if dateG.month == 12:
        startMonth = 1
        startYear = dateG.year + 1
    else:
        startMonth = dateG.month + 1
        startYear = dateG.year
    return dt.datetime(startYear, startMonth, 1)


def discrete_approx(rate, timestep):
    return (1 - (1 - rate)**(1 / timestep))
