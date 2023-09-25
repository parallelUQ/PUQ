import numpy as np
from reporting.report_pdf import generate_report,generate_report_tier
from collections import defaultdict
import csv
import datetime as dt

def build_report(instance_name,
                 instance,
                 policy,
                 profiles,
                 IHD_PLOT_FILE,
                 IYIH_PLOT_FILE,
                 n_replicas=300,
                 to_email=None,
                 config=None,
                 **kwargs):
    '''
        Gathers data to build a city report.
    '''
    report_data = {'instance_name': instance_name}
    report_data['CITY'] = config['city']
    #report_data['CITY'] = 'Austin'
    
    T = kwargs['T']
    cal = instance.cal
    population = instance.N.sum()
    interventions = kwargs['interventions']
    hosp_beds = kwargs['hosp_beds']
    policy_params = kwargs['policy_params']
    
    report_data['START-DATE'] = cal.calendar[0].strftime("%Y-%m-%d")
    report_data['END-DATE'] = cal.calendar[T - 1].strftime("%Y-%m-%d")
    report_data['policy_params'] = policy_params
    
    report_data['IHD_PLOT_FILE'] = IHD_PLOT_FILE
    report_data['IYIH_PLOT_FILE'] = IYIH_PLOT_FILE
    
    lb_band = 5
    ub_band = 95
    # Transform data of interest
    states_to_plot = ['S', 'E', 'IH', 'IA', 'IY', 'R', 'D', 'IYIH']
    states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot}
    states_ts['z'] = np.vstack(list(p['z'][:T] for p in profiles))
    states_ts['tier_history'] = np.vstack(list(p['tier_history'][:T] for p in profiles))
    
    central_path = 0
    mean_st = {v: states_ts[v][central_path] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
    min_st = {
        v: np.percentile(states_ts[v], q=lb_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    max_st = {
        v: np.percentile(states_ts[v], q=ub_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    
    central_path = 0
    mean_st = {v: states_ts[v][central_path] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
    min_st = {
        v: np.percentile(states_ts[v], q=5, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    max_st = {
        v: np.percentile(states_ts[v], q=95, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    # People that arrive above capacity
    # np.mean(np.sum(states_ts['IYIH']*(states_ts['IH']>=3239) , 1))
    
    # Stats
    all_states = ['S', 'E', 'IH', 'IA', 'IY', 'R', 'D', 'IYIH']
    all_states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in all_states}
    assert len(all_states_ts['IH']) >= n_replicas
    for v in all_states_ts:
        all_states_ts[v] = all_states_ts[v][:n_replicas]
    assert len(all_states_ts['IH']) == n_replicas
    # Hospitalizations Report
    # Probabilities of reaching x% of the capacity
    prob50 = np.round(
        np.sum(np.any(all_states_ts['IH'] >= 0.5 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob60 = np.round(
        np.sum(np.any(all_states_ts['IH'] >= 0.6 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob70 = np.round(
        np.sum(np.any(all_states_ts['IH'] >= 0.7 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob80 = np.round(
        np.sum(np.any(all_states_ts['IH'] >= 0.8 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob90 = np.round(
        np.sum(np.any(all_states_ts['IH'] >= 0.9 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob100 = np.round(
        np.sum(np.any(all_states_ts['IH'] >= 1 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob110 = np.round(
        np.sum(np.any(all_states_ts['IH'] >= 1.1 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    n_replicas_used = len(all_states_ts['IH'])
    print(f"{'P 50':10s}{'P 60':10s}{'P 70':10s}{'P 80':10s}{'P 90':10s}{'P 100':10s}{'P 110':10s}{'Scenarios':10s}")
    print(
        f"{prob50:<10.4f}{prob60:<10.4f}{prob70:<10.4f}{prob80:<10.4f}{prob90:<10.4f}{prob100:<10.4f}{prob110:<10.4f}{n_replicas_used}"
    )
    
    hosp_probs = {
        'HOSP-P60': prob60,
        'HOSP-P70': prob70,
        'HOSP-P80': prob80,
        'HOSP-P90': prob90,
        'HOSP-P100': prob100,
    }
    report_data.update(hosp_probs)
    
    # Min, Med, Max at the peak
    print('Hospitalization Peaks')
    peaks_vals = {}
    peaks_dates = {}
    peak_days = np.argmax(all_states_ts['IH'], axis=1)
    peak_vals = np.take_along_axis(all_states_ts['IH'], peak_days[:, None], axis=1)
    print(f'{"Percentile (%)":<15s} {"Peak IH":<15s}  {"Date":15}')
    for q in [0, 5, 10, 50, 90, 95, 100]:
        peak_day_percentile = int(np.percentile(peak_days, q))
        peak_percentile = np.percentile(peak_vals, q)
        peaks_vals[f'HOSP-PEAK-P{q}'] = int(peak_percentile)
        peaks_dates[f'HOSP-PEAK-DATE-P{q}'] = cal.calendar[peak_day_percentile].strftime("%Y-%m-%d")
        print(f'{q:<15} {peak_percentile:<15.0f}  {str(cal.calendar[peak_day_percentile])}')
    
    report_data.update(peaks_vals)
    report_data.update(peaks_dates)
    
    # Patients after capacity
    patients_excess = np.sum(all_states_ts['IYIH'] * (all_states_ts['IH'][:, :-1] >= kwargs['hosp_beds']), axis=1)
    report_data['UNSERVED-MEAN'] = np.round(patients_excess.mean())
    report_data['UNSERVED-SD'] = np.round(patients_excess.std())
    for q in [90, 95, 99, 100]:
        report_data[f'UNSERVED-P{q}'] = np.round(np.percentile(patients_excess, q))
    
    # Deaths
    all_states_ts_ind = {
        v: np.array(list(p[v][:T, :, :] for p in profiles if np.sum(p['IH']) > 100))
        for v in all_states
    }
    assert len(all_states_ts_ind['IH']) >= n_replicas
    for v in all_states_ts:
        all_states_ts_ind[v] = all_states_ts_ind[v][:n_replicas]
    assert len(all_states_ts_ind['IH']) == n_replicas
    # Deaths data
    
    avg_deaths_by_group = np.round(np.mean(all_states_ts_ind['D'][:, -1, :, :], axis=0).reshape((10, 1)), 0)
    Median_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), 50))
    CI5_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), 5))
    CI95_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), 95))
    deaths_report = {
        'MEAN-DEATHS': int(avg_deaths_by_group.sum()),
        'MEDIAN-DEATHS': int(Median_deaths),
        'P5-DEATHS': int(CI5_deaths),
        'P95-DEATHS': int(CI95_deaths),
    }
    print('Deaths End Horizon')
    print(f'Point forecast {all_states_ts["D"][0][-1]}')
    print(f'Mean {avg_deaths_by_group.sum()} Median:{Median_deaths} CI_5_95:[{CI5_deaths}-{CI95_deaths}]')
    print('Fraction by Age and Risk Group (1-5, L-H)')
    frac_age_risk = 100 * avg_deaths_by_group.reshape(5, 2) / avg_deaths_by_group.sum()
    print(frac_age_risk)
    for ag in range(len(frac_age_risk)):
        for rg in range(len(frac_age_risk[0])):
            deaths_report[f'DEATHS-A{ag}-R{rg}'] = f'{frac_age_risk[ag, rg]:.2f}'
    report_data.update(deaths_report)
    
    R_mean = np.mean(all_states_ts['R'][:, -1] / population)
    print(f'R End Horizon {R_mean}')
    # Policy
    #lockdown_threshold = ''
    #fdmi = policy_params['first_day_month_index']
    #policy = {(m, y): lockdown_threshold[fdmi[m, y]] for (m, y) in fdmi if fdmi[m, y] < T}
    print('Lockdown Threshold:')
    print('policy')
    report_data['policy'] = policy
    
    # Plot school closure and cocooning
    z_ts = mean_st['z'][central_path][:T]
    sc_co = [interventions[k].school_closure for k in z_ts]
    unique_policies = set(sc_co)
    sd_levels = [interventions[k].social_distance for k in z_ts]
    unique_sd_policies = set(sd_levels)
    
    intervals = {u: [False for t in range(len(z_ts) + 1)] for u in unique_policies}
    intervals_sd = {u: [False for t in range(len(z_ts) + 1)] for u in unique_sd_policies}
    for t in range(len(z_ts)):
        sc_co_t = interventions[z_ts[t]].school_closure
        for u in unique_policies:
            if u == sc_co_t:
                intervals[u][t] = True
                intervals[u][t + 1] = True
        for u_sd in unique_sd_policies:
            if u_sd == interventions[z_ts[t]].social_distance:
                intervals_sd[u_sd][t] = True
                intervals_sd[u_sd][t + 1] = True
    
    report_data['INITIAL-LOCKDOWN-KAPPA'] = 'XX\\%'
    report_data['LOCKDOWN-KAPPA'] = 'XX\\%'
    report_data['RELAXATION-KAPPA'] = 'XX\\%'
    
    # Plot social distance
    social_distance = [interventions[k].social_distance for k in z_ts]
    #policy_ax.plot(social_distance, c='k', alpha=0.6 * hide)  # marker='_', linestyle='None',
    hsd = np.sum(np.array(social_distance[:T]) == policy.tiers[-1]['transmission_reduction'])
    print(f'HIGH SOCIAL DISTANCE')
    print(f'Point Forecast: {hsd}')
    hsd_list = np.array([
        np.sum(
            np.array([interventions[k].social_distance for k in z_ts]) == policy.tiers[-1]['transmission_reduction'])
        for z_ts in states_ts['z']
    ])
    
    lockdown_report = {
        'MEAN-LOCKDOWN': f'{np.mean(hsd_list):.2f}',
        'MEDIAN-LOCKDOWN': f'{np.percentile(hsd_list,q=50)}',
        'P5-LOCKDOWN': f'{np.percentile(hsd_list,q=5)}',
        'P95-LOCKDOWN': f'{np.percentile(hsd_list,q=95)}'
    }
    report_data.update(lockdown_report)
    report_data['PATHS-IN-LOCKDOWN'] = 100*round(sum(hsd_list > 0)/len(hsd_list), 2)
    
    count_lockdowns = defaultdict(int)
    for z_ts in states_ts['z']:
        n_lockdowns = 0
        for ix_k in range(1, len(z_ts)):
            if interventions[z_ts[ix_k]].social_distance - interventions[z_ts[ix_k - 1]].social_distance > 0:
                n_lockdowns += 1
        count_lockdowns[n_lockdowns] += 1
    
    print(
        f'Mean: {np.mean(hsd_list):.2f} Median: {np.percentile(hsd_list,q=50)}   -  SD CI_5_95: {np.percentile(hsd_list,q=5)}-{np.percentile(hsd_list,q=95)}'
    )
    
    for nlock in count_lockdowns:
        print(f'Prob of having exactly {nlock} lockdowns: {count_lockdowns[nlock]/len(states_ts["z"]):4f}')
    unique_social_distance = np.unique(social_distance)
    # for usd in unique_social_distance:
    #     if usd > 0:
    #         offset = {0.1: -0.03, 0.2: -0.03, 0.4: -0.03, 0.6: -0.03, 0.8: -0.03, 0.9: 0.02}[usd]
    # policy_ax.annotate(f'{int(usd*100)}% social distance', (0.07, usd + offset),
    #                    xycoords='axes fraction',
    #                    color='k',
    #                    annotation_clip=True)  #
    
    generate_report(report_data, to_email)

def build_report_tiers(instance_name,
                 instance,
                 policy,
                 profiles,
                 IHT_PLOT_FILE,
                 ToIHT_PLOT_FILE,
                 ICU_PLOT_FILE,
                 ToICU_PLOT_FILE,
                 n_replicas=300,
                 to_email=None,
                 config=None,
                 stat_start=None,
                 template_file = "report_template_tier.tex",
                 **kwargs):
    '''
        Gathers data to build a city report.
    '''
    report_data = {'instance_name': instance_name}
    report_data['CITY'] = config['city']
    #report_data['CITY'] = 'Austin'
    
    T = kwargs['T']
    cal = instance.cal
    population = instance.N.sum()
    interventions = kwargs['interventions']
    hosp_beds = kwargs['hosp_beds']
    policy_params = kwargs['policy_params']
    
    report_data['START-DATE'] = cal.calendar[0].strftime("%Y-%m-%d")
    if stat_start == None:
        report_data['STATISTICS-START-DATE'] = report_data['START-DATE']
        T_start = 0
    else:
        report_data['STATISTICS-START-DATE'] = stat_start.strftime("%Y-%m-%d")
        T_start = instance.cal.calendar_ix[stat_start]
    report_data['END-DATE'] = cal.calendar[T - 1].strftime("%Y-%m-%d")
    report_data['policy_params'] = policy_params
    
    report_data['IHT_PLOT_FILE'] = IHT_PLOT_FILE
    report_data['ToIHT_PLOT_FILE'] = ToIHT_PLOT_FILE
    report_data['ICU_PLOT_FILE'] = ICU_PLOT_FILE
    report_data['ToICU_PLOT_FILE'] = ToICU_PLOT_FILE

    
    lb_band = 5
    ub_band = 95
    # Transform data of interest
    states_to_plot = ['S', 'E', 'IH', 'IA', 'IY', 'R', 'D', 'IYIH', 'IHT', 'ICU', 'ToIHT', 'ToICU']
    states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[T_start:T] for p in profiles)) for v in states_to_plot}
    states_ts['z'] = np.vstack(list(p['z'][T_start:T] for p in profiles))
    states_ts['tier_history'] = np.vstack(list(p['tier_history'][T_start:T] for p in profiles))
    
    central_path = 0
    mean_st = {v: states_ts[v][central_path] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
    # min_st = {
    #     v: np.percentile(states_ts[v], q=lb_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
    #     for v in states_ts
    # }
    # max_st = {
    #     v: np.percentile(states_ts[v], q=ub_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
    #     for v in states_ts
    # }
    # People that arrive above capacity
    # np.mean(np.sum(states_ts['IYIH']*(states_ts['IH']>=3239) , 1))
    
    # Stats
    all_states = ['S', 'E', 'IH', 'IA', 'IY', 'R', 'D', 'IYIH', 'IHT', 'ICU', 'ToIHT', 'ToICU']
    all_states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[T_start:T] for p in profiles)) for v in all_states}
    assert len(all_states_ts['IHT']) >= n_replicas
    for v in all_states_ts:
        all_states_ts[v] = all_states_ts[v][:n_replicas]
    assert len(all_states_ts['IHT']) == n_replicas
    # Hospitalizations Report
    # Probabilities of reaching x% of the capacity
    prob50 = np.round(
        np.sum(np.any(all_states_ts['IHT'] >= 0.5 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob60 = np.round(
        np.sum(np.any(all_states_ts['IHT'] >= 0.6 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob70 = np.round(
        np.sum(np.any(all_states_ts['IHT'] >= 0.7 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob80 = np.round(
        np.sum(np.any(all_states_ts['IHT'] >= 0.8 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob90 = np.round(
        np.sum(np.any(all_states_ts['IHT'] >= 0.9 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob100 = np.round(
        np.sum(np.any(all_states_ts['IHT'] >= 1 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    prob110 = np.round(
        np.sum(np.any(all_states_ts['IHT'] >= 1.1 * kwargs['hosp_beds'], axis=1)) / len(all_states_ts['IH']), 4)
    n_replicas_used = len(all_states_ts['IHT'])
    print(f"{'P 50':10s}{'P 60':10s}{'P 70':10s}{'P 80':10s}{'P 90':10s}{'P 100':10s}{'P 110':10s}{'Scenarios':10s}")
    print(
        f"{prob50:<10.4f}{prob60:<10.4f}{prob70:<10.4f}{prob80:<10.4f}{prob90:<10.4f}{prob100:<10.4f}{prob110:<10.4f}{n_replicas_used}"
    )
    
    hosp_probs = {
        'HOSP-P60': prob60,
        'HOSP-P70': prob70,
        'HOSP-P80': prob80,
        'HOSP-P90': prob90,
        'HOSP-P100': prob100,
    }
    report_data.update(hosp_probs)
        
    # Min, Med, Max at the peak
    print('Hospitalization Peaks')
    hosp_peaks_vals = {}
    hosp_peaks_dates = {}
    icu_peaks_vals = {}
    icu_peaks_dates = {}
    hosp_peak_days = np.argmax(all_states_ts['IHT'], axis=1)
    hosp_peak_vals = np.take_along_axis(all_states_ts['IHT'], hosp_peak_days[:, None], axis=1)
    icu_peak_days = np.argmax(all_states_ts['ICU'], axis=1)
    icu_peak_vals = np.take_along_axis(all_states_ts['ICU'], icu_peak_days[:, None], axis=1)
    print(f'{"Percentile (%)":<15s} {"Peak IHT":<15s}  {"Date":15}')
    
    hosp_peak_mean = np.mean(hosp_peak_vals)
    report_data['MEAN-HOSP-PEAK'] = np.round(hosp_peak_mean)
    icu_peak_mean = np.mean(icu_peak_vals)
    report_data['MEAN-ICU-PEAK'] = np.round(icu_peak_mean)
    # hosp_peak_std = np.std(hosp_peak_vals)
    # icu_peak_std = np.std(icu_peak_vals)
    # report_data['HOSP-PEAK-PPL'] = np.round(np.maximum(hosp_peak_mean - 1.645*hosp_peak_std,0))
    # report_data['HOSP-PEAK-PPH'] = np.round(hosp_peak_mean + 1.645*hosp_peak_std)
    # report_data['ICU-PEAK-PPL'] = np.round(np.maximum(icu_peak_mean - 1.645*icu_peak_std,0))
    # report_data['ICU-PEAK-PPH'] = np.round(icu_peak_mean + 1.645*icu_peak_std)
    report_data['HOSP-PEAK-PPL'] = np.round(np.percentile(hosp_peak_vals,lb_band))
    report_data['HOSP-PEAK-PPH'] = np.round(np.percentile(hosp_peak_vals,ub_band))
    report_data['ICU-PEAK-PPL'] = np.round(np.percentile(icu_peak_vals,lb_band))
    report_data['ICU-PEAK-PPH'] = np.round(np.percentile(icu_peak_vals,ub_band))


    
    for q in [50, 95, 100]:
        
        hosp_peak_day_percentile = int(np.round(np.percentile(hosp_peak_days, q)))
        hosp_peak_percentile = np.percentile(hosp_peak_vals, q)
        icu_peak_day_percentile = int(np.round(np.percentile(icu_peak_days, q)))
        icu_peak_percentile = np.percentile(icu_peak_vals, q)

        hosp_peaks_vals[f'HOSP-PEAK-P{q}'] = np.round(hosp_peak_percentile)
        hosp_peaks_dates[f'HOSP-PEAK-DATE-P{q}'] = cal.calendar[hosp_peak_day_percentile].strftime("%Y-%m-%d")
        icu_peaks_vals[f'ICU-PEAK-P{q}'] = np.round(icu_peak_percentile)
        icu_peaks_dates[f'ICU-PEAK-DATE-P{q}'] = cal.calendar[icu_peak_day_percentile].strftime("%Y-%m-%d")

        print(f'{q:<15} {hosp_peak_percentile:<15.0f}  {str(cal.calendar[hosp_peak_day_percentile])}')
        print(f'{q:<15} {icu_peak_percentile:<15.0f}  {str(cal.calendar[icu_peak_day_percentile])}')
    
    report_data.update(hosp_peaks_vals)
    report_data.update(hosp_peaks_dates)
    report_data.update(icu_peaks_vals)
    report_data.update(icu_peaks_dates)
    
    # Patients after capacity
    patients_excess = np.sum(np.maximum(all_states_ts['IHT'][:, :-1] - kwargs['hosp_beds'],0),axis = 1)
    report_data['PATHS-HOSP-UNMET'] = 100*np.round(np.sum(patients_excess > 0)/n_replicas,3)
    report_data['MEAN-HOSP-UNSERVED'] = np.round(patients_excess.mean())
    report_data['SD-HOSP-UNSERVED'] = np.round(patients_excess.std())
    # report_data['HOSP-UNSERVED-PPL'] = np.round(np.maximum(report_data['MEAN-HOSP-UNSERVED'] - 1.645*report_data['SD-HOSP-UNSERVED'],0),3)
    # report_data['HOSP-UNSERVED-PPH'] = np.round(report_data['MEAN-HOSP-UNSERVED'] + 1.645*report_data['SD-HOSP-UNSERVED'],3)
    report_data['HOSP-UNSERVED-PPL'] = np.round(np.percentile(patients_excess, lb_band),3)
    report_data['HOSP-UNSERVED-PPH'] = np.round(np.percentile(patients_excess, ub_band),3)

    for q in [50, 95, 100]:
        report_data[f'HOSP-UNSERVED-P{q}'] = np.round(np.percentile(patients_excess, q))
        
    icu_patients_excess = np.sum(np.maximum(all_states_ts['ICU'][:, :-1] - kwargs['icu_beds'],0),axis = 1)
    report_data['PATHS-ICU-UNMET'] = 100*np.round(np.sum(icu_patients_excess > 0)/n_replicas,3)
    report_data['MEAN-ICU-UNSERVED'] = np.round(icu_patients_excess.mean())
    report_data['SD-ICU-UNSERVED'] = np.round(icu_patients_excess.std())
    # report_data['ICU-UNSERVED-PPL'] = np.round(np.maximum(report_data['MEAN-ICU-UNSERVED'] - 1.645*report_data['SD-ICU-UNSERVED'],0),3)
    # report_data['ICU-UNSERVED-PPH'] = np.round(report_data['MEAN-ICU-UNSERVED'] + 1.645*report_data['SD-ICU-UNSERVED'],3)
    report_data['ICU-UNSERVED-PPL'] = np.round(np.percentile(icu_patients_excess, lb_band),3)
    report_data['ICU-UNSERVED-PPH'] = np.round(np.percentile(icu_patients_excess, ub_band),3)
    for q in [50, 95, 100]:
        report_data[f'ICU-UNSERVED-P{q}'] = np.round(np.percentile(icu_patients_excess, q))
    
    # Deaths
    all_states_ts_ind = {
        v: np.array(list(p[v][(T_start-1):T, :, :] for p in profiles if np.sum(p['IHT']) > 100))
        for v in all_states
    }
    assert len(all_states_ts_ind['IHT']) >= n_replicas
    for v in all_states_ts:
        all_states_ts_ind[v] = all_states_ts_ind[v][:n_replicas]
    assert len(all_states_ts_ind['IHT']) == n_replicas
    # Deaths data
    
    avg_deaths_by_group = np.round(np.mean(all_states_ts_ind['D'][:, -1, :, :] - all_states_ts_ind['D'][:, 0, :, :], axis=0).reshape((10, 1)), 0)
    P5_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :] - all_states_ts_ind['D'][:, 0, :, :], axis=(1, 2)), 5))
    P50_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :] - all_states_ts_ind['D'][:, 0, :, :], axis=(1, 2)), 50))
    P95_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :] - all_states_ts_ind['D'][:, 0, :, :], axis=(1, 2)), 95))
    # deaths_pph = np.round(np.mean(np.sum(all_states_ts_ind['D'][:, -1, :, :] - all_states_ts_ind['D'][:, 0, :, :], axis=(1, 2))) + 1.645*np.std(np.sum(all_states_ts_ind['D'][:, -1, :, :] - all_states_ts_ind['D'][:, 0, :, :], axis=(1, 2))))
    # deaths_ppl = np.round(np.maximum(np.mean(np.sum(all_states_ts_ind['D'][:, -1, :, :] - all_states_ts_ind['D'][:, 0, :, :], axis=(1, 2))) - 1.645*np.std(np.sum(all_states_ts_ind['D'][:, -1, :, :] - all_states_ts_ind['D'][:, 0, :, :], axis=(1, 2))),0))
    deaths_pph = P95_deaths
    deaths_ppl = P5_deaths
    deaths_report = {
        'MEAN-DEATHS': int(avg_deaths_by_group.sum()),
        'P50-DEATHS': int(P50_deaths),
        'P95-DEATHS': int(P95_deaths),
        'DEATHS-PPL': deaths_ppl,
        'DEATHS-PPH': deaths_pph
    }
    print('Deaths End Horizon')
    print(f'Point forecast {all_states_ts["D"][0][-1]}')
    print(f'Mean {avg_deaths_by_group.sum()} Median:{P50_deaths} CI_5_95:[{deaths_ppl}-{deaths_pph}]')
    print('Fraction by Age and Risk Group (1-5, L-H)')
    frac_age_risk = 100 * avg_deaths_by_group.reshape(5, 2) / avg_deaths_by_group.sum()
    print(frac_age_risk)
    for ag in range(len(frac_age_risk)):
        for rg in range(len(frac_age_risk[0])):
            deaths_report[f'DEATHS-A{ag}-R{rg}'] = f'{frac_age_risk[ag, rg]:.2f}'
    report_data.update(deaths_report)
    
    R_mean = np.mean(all_states_ts['R'][:, -1] - all_states_ts['R'][:, T_start] / population)
    print(f'R End Horizon {R_mean}')
    # Policy
    #lockdown_threshold = ''
    #fdmi = policy_params['first_day_month_index']
    #policy = {(m, y): lockdown_threshold[fdmi[m, y]] for (m, y) in fdmi if fdmi[m, y] < T}
    print('Lockdown Threshold:')
    print('policy')
    report_data['policy'] = policy
    
    # Plot school closure and cocooning
    z_ts = mean_st['z'][central_path][:]
    sc_co = [interventions[k].school_closure for k in z_ts]
    unique_policies = set(sc_co)
    sd_levels = [interventions[k].social_distance for k in z_ts]
    unique_sd_policies = set(sd_levels)
    
    intervals = {u: [False for t in range(len(z_ts) + 1)] for u in unique_policies}
    intervals_sd = {u: [False for t in range(len(z_ts) + 1)] for u in unique_sd_policies}
    for t in range(len(z_ts)):
        sc_co_t = interventions[z_ts[t]].school_closure
        for u in unique_policies:
            if u == sc_co_t:
                intervals[u][t] = True
                intervals[u][t + 1] = True
        for u_sd in unique_sd_policies:
            if u_sd == interventions[z_ts[t]].social_distance:
                intervals_sd[u_sd][t] = True
                intervals_sd[u_sd][t + 1] = True
    
    report_data['INITIAL-LOCKDOWN-KAPPA'] = 'XX\\%'
    report_data['LOCKDOWN-KAPPA'] = 'XX\\%'
    report_data['RELAXATION-KAPPA'] = 'XX\\%'
    
    # Plot social distance
    social_distance = [interventions[k].social_distance for k in z_ts]
    #policy_ax.plot(social_distance, c='k', alpha=0.6 * hide)  # marker='_', linestyle='None',
    hsd = np.sum(np.array(social_distance[:]) == policy.tiers[-1]['transmission_reduction'])
    print(f'HIGH SOCIAL DISTANCE')
    print(f'Point Forecast: {hsd}')
    hsd_list = np.array([
        np.sum(
            np.array([interventions[k].social_distance for k in z_ts]) == policy.tiers[-1]['transmission_reduction'])
        for z_ts in states_ts['z']
    ])
    
    lockdown_report = {
        'MEAN-LOCKDOWN': f'{np.mean(hsd_list):.2f}',
        'P50-LOCKDOWN': f'{np.percentile(hsd_list,q=50)}',
        'P95-LOCKDOWN': f'{np.percentile(hsd_list,q=95)}',
        # 'LOCKDOWN-PPH': f'{np.mean(hsd_list)+1.645*np.std(hsd_list):.2f}',
        # 'LOCKDOWN-PPL': f'{np.maximum(np.mean(hsd_list)-1.645*np.std(hsd_list),0):.2f}',
        'LOCKDOWN-PPH': f'{np.percentile(hsd_list,q=95):.2f}',
        'LOCKDOWN-PPL': f'{np.percentile(hsd_list,q=5):.2f}',
    }
    report_data.update(lockdown_report)
    report_data['PATHS-IN-LOCKDOWN'] = 100*round(sum(hsd_list > 0)/len(hsd_list), 4)
    
    count_lockdowns = defaultdict(int)
    for z_ts in states_ts['z']:
        n_lockdowns = 0
        for ix_k in range(1, len(z_ts)):
            if interventions[z_ts[ix_k]].social_distance - interventions[z_ts[ix_k - 1]].social_distance > 0:
                n_lockdowns += 1
        count_lockdowns[n_lockdowns] += 1
    
    print(
        f"Mean: {np.mean(hsd_list):.2f} Median: {np.percentile(hsd_list,q=50)}   -  SD CI_5_95: {report_data['LOCKDOWN-PPL']}-{report_data['LOCKDOWN-PPH']}"
    )
    
    for nlock in np.sort(list(count_lockdowns.keys())):
        print(f'Prob of having exactly {nlock} stricter tier change: {count_lockdowns[nlock]/len(states_ts["z"]):4f}')
    unique_social_distance = np.unique(social_distance)
    # for usd in unique_social_distance:
    #     if usd > 0:
    #         offset = {0.1: -0.03, 0.2: -0.03, 0.4: -0.03, 0.6: -0.03, 0.8: -0.03, 0.9: 0.02}[usd]
    # policy_ax.annotate(f'{int(usd*100)}% social distance', (0.07, usd + offset),
    #                    xycoords='axes fraction',
    #                    color='k',
    #                    annotation_clip=True)  #
    
    generate_report_tier(report_data,template_file = template_file, to_email = to_email)


def csv_UT_Style(stoch_replicas, out_field, out_file, start_date, percentileList=[2.5, 97.5]):
    # output the simulation data, in the format of TACC team
    
    # obtain the output based on the out_field for each item of stoch_replicas
    outList = []
    for i in range(len(stoch_replicas)):
        outList.append(np.sum(stoch_replicas[i][out_field], axis=(1, 2)))
    
    # the number of time periods
    T = outList[0].shape[0]
    TList = [
        "{}/{}/{}".format((start_date + dt.timedelta(days=t)).month, (start_date + dt.timedelta(days=t)).day,
                          (start_date + dt.timedelta(days=t)).year) for t in range(T)
    ]
    
    # columns of day's index and date
    outTotal = [[t for t in range(T)], TList]
    titleRow = ['', 'date']
    
    # column of mean
    outMean = np.mean(outList, axis=0)
    outTotal.append(outMean)
    titleRow.append('sto_idx_0')
    
    # column of median
    outMedian = np.median(outList, axis=0)
    outTotal.append(outMedian)
    titleRow.append('median')
    
    # columns of percentile
    outPerc = {}
    for perc in percentileList:
        outPerc[perc] = np.percentile(outList, perc, axis=0)
        outTotal.append(outPerc[perc])
        if perc < 50:
            titleRow.append('lower_{}%'.format(perc))
        else:
            titleRow.append('upper_{}%'.format(perc))
    
    # columns of max, min
    outMax = np.max(outList, axis=0)
    outTotal.append(outMax)
    titleRow.append('max')
    
    outMin = np.min(outList, axis=0)
    outTotal.append(outMin)
    titleRow.append('min')
    
    # write the csv file
    fi = open(out_file, 'w', newline='')
    csvWriter = csv.writer(fi, dialect='excel')
    csvWriter.writerow(titleRow)
    csvWriter.writerows(np.transpose(outTotal))
    fi.close()