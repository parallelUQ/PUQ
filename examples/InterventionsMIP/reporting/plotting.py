'''
Module for plotting function
'''
import os
import sys
import numpy as np
import pandas as pd
import time
import argparse
import calendar as py_cal
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib import rc
import matplotlib.patches as patches
import matplotlib.colors as pltcolors
from collections import defaultdict
from utils import round_closest, roundup
from InterventionsMIP import plots_path, instances_path
import copy

plt.rcParams['hatch.linewidth'] = 3.0

colors = {'S': 'b', 'E': 'y', 'IA': 'c', 'IY': 'm', 'IH': 'k', 'R': 'g', 'D': 'k', 'ToIHT': 'teal', 'ICU': 'k', 'ToICU': 'teal', 'IHT': 'k', 'ITot': 'k'}
light_colors = {'IH':'silver','ToIHT':'paleturquoise', 'ICU':'silver', 'ToICU': 'paleturquoise', 'IHT': 'silver', 'ITot': 'silver'}
l_styles = {'sim': '-', 'opt': '--'}
compartment_names = {
    'ITot': 'Total Infectious',
    'IY': 'Symptomatic',
    'IH': 'General Beds',
    'ToIHT': 'COVID-19 Hospital Admissions\n(Seven-day Average)',
    'D': 'Deaths',
    'R': 'Recovered',
    'S': 'Susceptible',
    'ICU': 'COVID-19 ICU Patients',
    'IHT': 'COVID-19 Hospitalizations',
    'ToICU': 'Daily COVID-19 ICU Admissions'
}

def colorDecide(u,tier_by_tr):
    preCoded_color = ["blue","yellow","orange","red"]
    colorDict = {}
    for tKey in tier_by_tr.keys():
        colorDict[tier_by_tr[tKey]["color"]] = tKey
    if u < colorDict["blue"]:
        return "white",""
    else:
        # if it is a color above blue, forced to be below red
        belowTier = -1
        aboveTier = 2
        for item in preCoded_color:
            if (u > colorDict[item])and(colorDict[item] >= belowTier):
                belowTier = colorDict[item]
            if (u < colorDict[item])and(colorDict[item] <= aboveTier):
                aboveTier = colorDict[item]
        aboveColor = pltcolors.to_rgb(tier_by_tr[aboveTier]["color"])
        belowColor = pltcolors.to_rgb(tier_by_tr[belowTier]["color"])
        ratio = (u - belowTier)/(aboveTier - belowTier)
        setcolor = ratio*np.array(aboveColor) + (1-ratio)*np.array(belowColor)
        return setcolor,tier_by_tr[aboveTier]["color"]+\
                            "_"+tier_by_tr[belowTier]["color"]+\
                            "_"+str(ratio)

def find_central_path(city, states_to_plot_temp, states_ts_temp, real_hosp, real_icu, real_new_admission=None):
    '''
    Obtains the central path id

    Args:
        TO DO
    '''
    central_path_id = 0
    weights_obs = 0.1 #0.005
    weights = np.array(np.repeat((1 - weights_obs)/12, 12)) 
    data_metrics = np.empty((300, 0), float)
    for v_t in states_to_plot_temp:
        data_metrics = np.append(data_metrics, np.max(states_ts_temp[v_t], axis = 1, keepdims = True), 1)        
        data_metrics = np.append(data_metrics, np.argmax(states_ts_temp[v_t], axis = 1).reshape(len(states_ts_temp[v_t]), 1), 1)    
        data_metrics = np.append(data_metrics, np.quantile(states_ts_temp[v_t], 0.5, axis = 1, keepdims = True), 1)        
        data_metrics = np.append(data_metrics, np.sum(states_ts_temp[v_t], axis = 1, keepdims = True), 1)
    
    #Standardize
    std_data_metrics = (data_metrics - np.mean(data_metrics, axis=0)) / np.std(data_metrics, axis=0)
    errorlist1 = (np.square(std_data_metrics).dot(weights)).reshape(len(states_ts_temp['IHT']), 1)  
    if city == 'austin':
        w = 7.3*(1 - 0.10896) + 9.9*0.10896
        #Metric for deviations from the observed data
        if np.sum(real_hosp) > np.sum(real_icu):
            x_dev = np.mean(np.square((states_ts_temp['IHT'][:, 0:len(real_hosp)] - real_hosp[0:])), axis = 1, keepdims = True)
            z_dev = np.mean(np.square((states_ts_temp['ICU'][:, 0:len(real_icu)] - real_icu[0:])), axis = 1, keepdims = True)
        else:
            x_dev = np.mean(np.square((states_ts_temp['IHT'][:, 0:len(real_icu)] - real_icu[0:])), axis = 1, keepdims = True)
            z_dev = np.mean(np.square((states_ts_temp['ICU'][:, 0:len(real_hosp)] - real_hosp[0:])), axis = 1, keepdims = True)
        y_dev = np.mean(np.square((states_ts_temp['ToIHT'][:, 0:len(real_new_admission)] - real_new_admission[0:])), axis = 1, keepdims = True)
        errorlist2 = 1/(np.square(w))*x_dev + np.square(2.5)/(np.square(w))*z_dev + y_dev
    else:
        w = 7.3*(1 - 0.10896) + 9.9*0.10896
        #Metric for deviations from the observed data
        if np.sum(real_hosp) > np.sum(real_icu):
            x_dev = np.mean(np.square((states_ts_temp['IHT'][:, 0:len(real_hosp)] - real_hosp[0:])), axis = 1, keepdims = True)
            z_dev = np.mean(np.square((states_ts_temp['ICU'][:, 0:len(real_icu)] - real_icu[0:])), axis = 1, keepdims = True)
        else:
            x_dev = np.mean(np.square((states_ts_temp['IHT'][:, 0:len(real_icu)] - real_icu[0:])), axis = 1, keepdims = True)
            z_dev = np.mean(np.square((states_ts_temp['ICU'][:, 0:len(real_hosp)] - real_hosp[0:])), axis = 1, keepdims = True)
        errorlist2 = 1/(np.square(w))*x_dev + np.square(2.5)/(np.square(w))*z_dev
       
    errorlist = (errorlist1 + weights_obs*errorlist2).tolist()
    central_path_id = errorlist.index(min(errorlist))
    
    if central_path_id == 0:
        sorted_er = sorted(errorlist)
        central_path_id = errorlist.index(sorted_er[1])

    print("central_path_id: ", central_path_id)  
    return central_path_id

def change_avg(all_st, min_st ,max_st, mean_st, nday_avg):
    # obtain the n-day average of the statistics
    all_st_copy = copy.deepcopy(all_st)
    min_st_copy = copy.deepcopy(min_st)
    max_st_copy = copy.deepcopy(max_st)
    mean_st_copy = copy.deepcopy(mean_st)
    
    # change all statistics to n-day average
    for v in all_st_copy.keys():
        if v not in ['z', 'tier_history']:
            for i in range(len(all_st_copy[v])):
                for t in range(len(all_st_copy[v][i])):
                    all_st_copy[v][i][t] = np.mean(all_st[v][i][np.maximum(t-nday_avg,0):t+1])
            for t in range(len(min_st_copy[v])):
                min_st_copy[v][t] = np.mean(min_st[v][np.maximum(t-nday_avg,0):t+1])
            for t in range(len(max_st_copy[v])):
                max_st_copy[v][t] = np.mean(max_st[v][np.maximum(t-nday_avg,0):t+1])
            for t in range(len(mean_st_copy[v])):
                mean_st_copy[v][t] = np.mean(mean_st[v][np.maximum(t-nday_avg,0):t+1])
            
    return all_st_copy,min_st_copy,max_st_copy,mean_st_copy

def plot_multi_tier_sims(instance_name,
                         instance,
                         policy,
                         profiles,
                         profile_labels,
                         real_hosp,
                         plot_left_axis=['IH'],
                         plot_right_axis=[],
                         scale_plot=False,
                         align_axes=True,
                         show=True,
                         plot_triggers=False,
                         plot_trigger_annotations=False,
                         plot_legend=False,
                         y_lim=None,
                         n_replicas=300,
                         config=None,
                         hosp_beds_list=None,
                         real_new_admission=None,
                         real_hosp_or_icu=None,
                         bed_scale=1,
                         is_representative_path=False,
                         t_start = -1,
                         central_path_id=0,
                         cap_path_id=0,
                         vertical_fill=True,
                         nday_avg=None,
                         **kwargs):
    '''
    Plots a list of profiles in the same figure. Each profile corresponds
    to a stochastic replica for the given instance.

    Args:
        profiles (list of dict): a list of dictionaries that contain epi vars profiles
        profile_labels (list of str): name of each profile
        plot_only (list of str): list of variable names to be plot
    '''
    plt.rcParams["font.size"] = "18"
    T = kwargs['T']
    if "add_tiers" in kwargs.keys():
        add_tiers = kwargs["add_tiers"]
    cal = instance.cal
    population = instance.N.sum()
    interventions = kwargs['interventions']
    policy_params = kwargs['policy_params']
    if hosp_beds_list is None:
        hosp_beds_list = [instance.hosp_beds]
    hosp_beds = hosp_beds_list[0]
    
    lb_band = 5
    ub_band = 95
    
    text_size = 28
    fig, (ax1, actions_ax) = plt.subplots(2, 1, figsize=(17, 9), gridspec_kw={'height_ratios': [10, 1.1]})
    # Main axis
    # ax1.set_xlabel('Time')
    ax2 = None
    # Policy axis
    policy_ax = ax1.twinx()
    #policy_ax.set_ylabel('Social Distance')
    # If there are plot to be on the right axis, move policy_ax
    # Second, show the right spine.
    if len(plot_right_axis) > 0:
        # Create second axis
        ax2 = ax1.twinx()
        # Fix policy axis
        policy_ax.spines["right"].set_position(("axes", 1.1))
        make_patch_spines_invisible(policy_ax)
        policy_ax.spines["right"].set_visible(True)
    
    # Start plots
    max_y_lim_1 = population if 'S' in plot_left_axis or 'R' in plot_left_axis else 0
    max_y_lim_2 = population if 'S' in plot_right_axis or 'R' in plot_right_axis else 0
    plotted_lines = []
    
    # Add IHT field
    if 'ICU' in profiles[0].keys():
        for p in profiles:
            p['IHT'] = p['IH'] + p['ICU']
    
    # Transform data of interest
    states_to_plot = plot_left_axis + plot_right_axis
    last_day_hosp_data = len(real_hosp) - 1
    lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
    ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
    states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot}
    states_ts['z'] = np.vstack(list(p['z'][:T] for p in profiles))
    states_ts['tier_history'] = np.vstack(list(p['tier_history'][:T] for p in profiles))
    
    states_to_plot_temp = ['IHT','ToIHT', 'ICU']
    states_ts_temp = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot_temp}
    
    
    central_path = 0
    representative_path_id = 0
    print("Printed seed is: ", profiles[0]["seed"])

    if is_representative_path == False:
        central_path = central_path_id
        mean_st = {v: states_ts[v][central_path] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
    else:
        representative_path_id = find_central_path(instance.city, states_to_plot_temp, states_ts_temp, real_hosp, real_hosp_or_icu, real_new_admission)
        mean_st = {v: states_ts[v][representative_path_id] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
        central_path = representative_path_id
        cap_path_id = representative_path_id

    all_st = {v: states_ts[v][:] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
    min_st = {
        v: np.percentile(states_ts[v], q=lb_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    max_st = {
        v: np.percentile(states_ts[v], q=ub_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    if nday_avg is not None:
        all_st, min_st ,max_st, mean_st = change_avg(all_st, min_st ,max_st, mean_st, nday_avg)
    # People that arrive above capacity
    # np.mean(np.sum(states_ts['IYIH']*(states_ts['IH']>=3239) , 1))
    new_profiles = [mean_st, min_st, max_st]
    
    # Stats
    all_states = ['S', 'E', 'IH', 'IA', 'IY', 'R', 'D']
    if 'ICU' in profiles[0].keys():
        all_states.append('ICU')
        all_states.append('IHT')
        all_states.append('ToICU')
    all_states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in all_states}
    #assert len(all_states_ts['IH']) >= n_replicas
    for v in all_states_ts:
        all_states_ts[v] = all_states_ts[v][:n_replicas]
    #assert len(all_states_ts['IH']) == n_replicas
    # Hospitalizations Report
    # Probabilities of reaching x% of the capacity
    prob50 = np.sum(np.any(all_states_ts['IH'] >= 0.5 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob60 = np.sum(np.any(all_states_ts['IH'] >= 0.6 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob70 = np.sum(np.any(all_states_ts['IH'] >= 0.7 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob80 = np.sum(np.any(all_states_ts['IH'] >= 0.8 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob90 = np.sum(np.any(all_states_ts['IH'] >= 0.9 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob100 = np.sum(np.any(all_states_ts['IH'] >= 1 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob110 = np.sum(np.any(all_states_ts['IH'] >= 1.1 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    n_replicas_used = len(all_states_ts['IH'])
    print(f"{'P 50':10s}{'P 60':10s}{'P 70':10s}{'P 80':10s}{'P 90':10s}{'P 100':10s}{'P 110':10s}{'Scenarios':10s}")
    print(
        f"{prob50:<10.4f}{prob60:<10.4f}{prob70:<10.4f}{prob80:<10.4f}{prob90:<10.4f}{prob100:<10.4f}{prob110:<10.4f}{n_replicas_used}"
    )
    # Min, Med, Max at the peak
    print('Hospitalization Peaks')
    peak_days = np.argmax(all_states_ts['IH'], axis=1)
    peak_vals = np.take_along_axis(all_states_ts['IH'], peak_days[:, None], axis=1)
    print(f'{"Percentile (%)":<15s} {"Peak IH":<15s}  {"Date":15}')
    for q in [0, 5, 10, 50, 90, 100]:
        peak_day_percentile = int(np.percentile(peak_days, q))
        peak_percentile = np.percentile(peak_vals, q)
        print(f'{q:<15} {peak_percentile:<15.0f}  {str(cal.calendar[peak_day_percentile])}')
    
    # Deaths
    all_states_ts_ind = {
        v: np.array(list(p[v][:T, :, :] for p in profiles)) for v in all_states
        }
    
    #assert len(all_states_ts_ind['IH']) >= n_replicas
    for v in all_states_ts:
        all_states_ts_ind[v] = all_states_ts_ind[v][:n_replicas]
    #assert len(all_states_ts_ind['IH']) == n_replicas
    # Deaths data
    avg_deaths_by_group = np.round(np.mean(all_states_ts_ind['D'][:, -1, :, :], axis=0).reshape((10, 1)), 0)
    Median_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), 50))
    CI5_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), lb_band))
    CI95_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), ub_band))
    print('Deaths End Horizon')
    print(f'Point forecast {all_states_ts["D"][0][-1]}')
    print(f'Mean {avg_deaths_by_group.sum()} Median:{Median_deaths} CI_5_95:[{CI5_deaths}-{CI95_deaths}]')
    print('Fraction by Age and Risk Group (1-5, L-H)')
    print(100 * avg_deaths_by_group.reshape(5, 2) / avg_deaths_by_group.sum())
    R_mean = np.mean(all_states_ts['R'][:, -1] / population)
    print(f'R End Horizon {R_mean}')
    # Policy
    lockdown_threshold = policy.lockdown_thresholds[0]
    # fdmi = policy_params['first_day_month_index']
    # policy = {(m, y): lockdown_threshold[fdmi[m, y]] for (m, y) in fdmi if fdmi[m, y] < T}
    # print('Lockdown Threshold:')
    # print(policy)
    hide = 1
    l_style = l_styles['sim']
    for v in plot_left_axis:
        max_y_lim_1 = np.maximum(max_y_lim_1, np.max(max_st[v]))
        label_v = compartment_names[v]
        if v != 'IYIHa':
            v_a = ax1.plot(mean_st[v].T * bed_scale, c=colors[v], linestyle=l_style, linewidth=2, label=label_v, alpha=1 * hide, zorder = 50)
            plotted_lines.append(v_a[0])
            v_aa = ax1.plot(all_st[v].T * bed_scale, c=light_colors[v], linestyle=l_style, linewidth=1, label=label_v, alpha=0.8 * hide)
            plotted_lines.append(v_aa[0])
        #if central_path != 0:
        #    ax1.fill_between(range(len(max_st[v])),
        #                     max_st[v],
        #                     min_st[v],
        #                     color=colors[v],
        #                     linestyle=l_style,
        #                     facecolor="none",
        #                     linewidth=0.0,
        #                     alpha=0.5 * hide)
        if v == 'IH' or v == 'ICU' or v == 'IHT' or v == 'ITot':
            real_h_plot = ax1.scatter(range(len(real_hosp_or_icu)), real_hosp_or_icu, color='maroon', label='Actual hospitalizations',zorder=100,s=15)
            max_y_lim_1 = np.maximum(roundup(np.max(hosp_beds_list), 100), max_y_lim_1)
            try:
                if v == 'IH' or v == 'IHT':
                    ax1.plot(profiles[cap_path_id]['capacity'][:T], color='k', linestyle='-', linewidth=3)
                else:
                    for hosp_beds_lines in hosp_beds_list:
                        ax1.hlines(hosp_beds_lines, 0, T, color='k', linestyle='-', linewidth=3)
            except:
                for hosp_beds_lines in hosp_beds_list:
                    ax1.hlines(hosp_beds_lines, 0, T, color='k', linestyle='-', linewidth=3)
            xpos = 30  #440  #200  # 440
            if plot_trigger_annotations:
                ax1.annotate('Hospital capacity', (xpos, hosp_beds + 150),
                             xycoords='data',
                             color=colors[v],
                             annotation_clip=True,
                             fontsize=text_size + 2)  #
            if plot_triggers:
                ax1.hlines(policy_params['hosp_beds'] * 0.6, 0, T, 'b', '-', linewidth=3)
                for tier_ix, tier in enumerate(policy.tiers):
                    ax1.plot([policy.lockdown_thresholds[tier_ix][0]]*T, color=tier['color'], linewidth=5)
                    xpos = np.minimum(405, int(T * 0.65))  #180  #405
                    xytext = (xpos, lockdown_threshold[xpos] - 20)

                if plot_trigger_annotations:
                    ax1.annotate('Safety threshold', (xpos, policy_params['hosp_level_release'] - 250),
                                 xycoords='data',
                                 color='b',
                                 annotation_clip=True,
                                 fontsize=text_size + 2)
        if v == 'ToIHT' or v == 'ToICU':
            if v == 'ToIHT':
                if real_new_admission is not None:
                    real_h_plot = ax1.scatter(range(len(real_new_admission)), real_new_admission, color='maroon', label='New hospital admission',zorder=100,s=15)
            if plot_triggers and vertical_fill:
                #if central_path > 0:
                #    IYIH_mov_ave = []
                #    for t in range(T):
                #        IYIH_mov_ave.append(np.mean(mean_st[v][np.maximum(0, t - 7):t]))
                #    v_avg = ax1.plot(IYIH_mov_ave, c='black', linestyle=l_style, label=f'Moving Avg. {label_v}')
                # plotted_lines.append(v_avg[0])
                for tier_ix, tier in enumerate(policy.tiers):
                    ax1.plot([policy.lockdown_thresholds[tier_ix][0]]*T, color=tier['color'], linewidth=5)
                    xpos = np.minimum(405, int(T * 0.65))  #180  #405
                    xytext = (xpos, lockdown_threshold[xpos] - 20)
                    if plot_trigger_annotations:
                        ax1.annotate('Lock-down threshold',
                                     xy=(120, lockdown_threshold[120]),
                                     xytext=xytext,
                                     xycoords='data',
                                     textcoords='data',
                                     color='b',
                                     annotation_clip=True,
                                     fontsize=text_size + 2)
            if "plot_ACS_triggers" in kwargs.keys():
                if kwargs["plot_ACS_triggers"]:
                    ax1.plot([policy.acs_thrs]*T, color='k', linewidth=5)
    for v in plot_right_axis:
        max_y_lim_2 = np.maximum(max_y_lim_2, np.max(max_st[v]))
        label_v = compartment_names[v]
        v_a = ax2.plot(mean_st[v].T, c=colors[v], linestyle=l_style, label=label_v)
        plotted_lines.append(v_a[0])
        ax2.fill_between(range(T), min_st[v], max_st[v], color=colors[v], linestyle=l_style, alpha=0.5)
        if v == 'IH':
            max_y_lim_2 = np.maximum(roundup(hosp_beds, 100), max_y_lim_2)
            ax2.hlines(hosp_beds, 0, T, color='r', linestyle='--', label='N. of beds')
            if plot_triggers:
                ax2.hlines(policy_params['hosp_level_release'], 0, T, 'b', '--')
                ax2.annotate('Trigger - Current hospitalizations ',
                             (0.05, 0.78 * policy_params['hosp_level_release'] / max_y_lim_1),
                             xycoords='axes fraction',
                             color='b',
                             annotation_clip=True)
        if v == 'ToIHT':
            if plot_triggers:
                ax2.plot(lockdown_threshold[:T], 'b-')
                xytext = (160, lockdown_threshold[160] - 15)
                ax2.annotate('Trigger - Avg. Daily Hospitalization',
                             xy=(120, lockdown_threshold[120]),
                             xytext=xytext,
                             xycoords='data',
                             textcoords='data',
                             color='b',
                             annotation_clip=True)
                # ax2.annotate('                                        ',
                #              xy=(85, lockdown_threshold[85]),
                #              xytext=xytext,
                #              xycoords='data',
                #              textcoords='data',
                #              arrowprops={'arrowstyle': '-|>'},
                #              color='b',
                #              annotation_clip=True)
    
    # Plotting the policy
    # Plot school closure and cocooning
    tiers = policy.tiers
    z_ts = profiles[central_path]['z'][:T]
    tier_h = profiles[central_path]['tier_history'][:T]
    print('seed was', profiles[central_path]['seed'])
    sc_co = [interventions[k].school_closure for k in z_ts]
    unique_policies = set(sc_co)
    sd_lvl = [interventions[k].social_distance for k in z_ts]
    sd_levels = [tier['transmission_reduction'] for tier in tiers] + [0, 0.95] + sd_lvl
    unique_sd_policies = list(set(sd_levels))
    unique_sd_policies.sort()
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
    
    interval_color = {0: 'orange', 1: 'purple', 0.5: 'green'}
    interval_labels = {0: 'Schools Open', 1: 'Schools Closed', 0.5: 'Schools P. Open'}
    interval_alpha = {0: 0.3, 1: 0.3, 0.5: 0.3}
    for u in unique_policies:
        u_color = interval_color[u]
        u_label = interval_labels[u]
        
        actions_ax.fill_between(
            range(len(z_ts) + 1),
            0,
            1,
            where=intervals[u],
            color='white',  #u_color,
            alpha=0,  #interval_alpha[u],
            label=u_label,
            linewidth=0,
            hatch = '/',
            step='pre')
    # for kv in interval_labels:
    #     kv_label = interval_labels[kv]
    #     kv_color = interval_color[kv]
    #     kv_alpha = interval_alpha[kv]
    #     actions_ax.fill_between(range(len(z_ts) + 1),
    #                             0,
    #                             0.0001,
    #                             color=kv_color,
    #                             alpha=kv_alpha,
    #                             label=kv_label,
    #                             linewidth=0,
    #                             step='pre')
    
    sd_labels = {
        0: '',
        0.95: 'Initial lock-down',
    }
    sd_labels.update({tier['transmission_reduction']: tier['name'] for tier in tiers})
    tier_by_tr = {tier['transmission_reduction']: tier for tier in tiers}
    tier_by_tr[0.746873309820472] = {
        "name": 'Ini Lockdown',
        "transmission_reduction": 0.95,
        "cocooning": 0.95,
        "school_closure": 1,
        "min_enforcing_time": 0,
        "daily_cost": 0,
        "color": 'darkgrey'
    }
    
    if "add_tiers" in kwargs.keys():
        for add_t in add_tiers.keys():
            tier_by_tr[add_t] = {"color": add_tiers[add_t],
                                 "name": "added stage"}
    
    if align_axes:
        max_y_lim_1 = np.maximum(max_y_lim_1, max_y_lim_2)
        max_y_lim_2 = max_y_lim_1
    if y_lim is not None:
        max_y_lim_1 = y_lim
    else:
        max_y_lim_1 = roundup(max_y_lim_1, 100 if 'ToIHT' in plot_left_axis else 1000)
        
    if vertical_fill:
        for u in unique_sd_policies:
            try:
                if u in tier_by_tr.keys():
                    u_color = tier_by_tr[u]['color']
                    u_label = f'{tier_by_tr[u]["name"]}' if u > 0 else ""
                else:
                    u_color,u_label = colorDecide(u,tier_by_tr)
                u_alpha1 = 0.6
                u_alpha2 = 0.6
                fill_1 = intervals_sd[u].copy()
                fill_2 = intervals_sd[u].copy()
                for i in range(len(intervals_sd[u])):
                    if 'history_white' in kwargs.keys() and kwargs['history_white']:
                        if i <= t_start:
                            fill_2[i] = False
                        fill_1[i] = False
                    else:
                        if i <= t_start:
                            fill_2[i] = False
                        else:
                            fill_1[i] = False
                        
                policy_ax.fill_between(range(len(z_ts) + 1),
                                       0,
                                       1,
                                       where=fill_1,
                                       color=u_color,
                                       alpha=u_alpha1,
                                       label=u_label,
                                       linewidth=0.0,
                                       step='pre')
                policy_ax.fill_between(range(len(z_ts) + 1),
                                       0,
                                       1,
                                       where=fill_2,
                                       color=u_color,
                                       alpha=u_alpha2,
                                       label=u_label,
                                       linewidth=0.0,
                                       step='pre')
            except Exception:
                print(f'WARNING: TR value {u} was not plotted')
    else:
        # fill the horizontal policy color
        for ti in range(len(tiers)):
            u = tiers[ti]['transmission_reduction']
            if u in tier_by_tr.keys():
                u_color = tier_by_tr[u]['color']
                u_label = f'{tier_by_tr[u]["name"]}' if u > 0 else ""
            else:
                u_color,u_label = colorDecide(u,tier_by_tr)
            u_alpha = 0.6
            u_lb = policy.lockdown_thresholds[ti][0]
            u_ub = policy.lockdown_thresholds_ub[ti][0]
            if u_ub == np.inf:
                u_ub = max_y_lim_1
            
            if u_lb >= 0 and u_ub >= 0:
                policy_ax.fill_between(range(len(z_ts) + 1),
                                   u_lb/max_y_lim_1,
                                   u_ub/max_y_lim_1,
                                   color=u_color,
                                   alpha=u_alpha,
                                   label=u_label,
                                   linewidth=0.0,
                                   step='pre')

    if "acs_fill" in kwargs.keys():
        # fill the ACS plot
        policy_ax.fill_between(range(len(z_ts) + 1),
                                       0,
                                       1,
                                       color='white',
                                       linewidth=0.0,
                                       step='pre')
        policy_ax.fill_between(range(len(z_ts) + 1),
                                   0,
                                   hosp_beds/max_y_lim_1,
                                   color='lightgreen',
                                   alpha=0.6,
                                   linewidth=0.0,
                                   step='pre')
        fill_acs = [False]*(len(z_ts) + 1)
        acs_rec = -1
        acs_date = []
        for tind in range(T):
            if profiles[cap_path_id]['capacity'][tind] > hosp_beds:
                acs_date.append(tind)
                fill_acs[tind] = True
                acs_rec = profiles[cap_path_id]['capacity'][tind]
        ax1.plot(acs_date,[hosp_beds]*len(acs_date),color='gray', linestyle='-', linewidth=1)
        if acs_rec > 0:
            policy_ax.fill_between(range(len(z_ts) + 1),
                                   hosp_beds/max_y_lim_1,
                                   acs_rec/max_y_lim_1,
                                   where = fill_acs,
                                   color='forestgreen',
                                   alpha=0.6,
                                   linewidth=0.0,
                                   step='pre')
            
    # # Plot again for consolidated legend
    # for u in sd_alphas:
    #     u_label = sd_labels[u]
    #     policy_ax.fill_between(range(len(z_ts) + 1),
    #                            0,
    #                            0.0001,
    #                            color=u_color,
    #                            alpha=sd_alphas[u],
    #                            label=u_label,
    #                            linewidth=0,
    #                            step='pre')
    # Plot social distance
    social_distance = [interventions[k].social_distance for k in z_ts]
    #policy_ax.plot(social_distance, c='k', alpha=0.6 * hide)  # marker='_', linestyle='None',
    hsd = np.sum(np.array(social_distance[:T]) >= 0.78)
    print(f'HIGH SOCIAL DISTANCE')
    print(f'Point Forecast: {hsd}')
    hsd_list = np.array(
        [np.sum(np.array([interventions[k].social_distance for k in z_ts]) >= 0.78) for z_ts in states_ts['z']])
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
    
    # START PLOT STYLING
    # Axis limits
    ax1.set_ylim(0, max_y_lim_1)
    if ax2 is not None:
        ax2.set_ylim(0, roundup(max_y_lim_2, 1000))
    policy_ax.set_ylim(0, 1)
    
    # plot a vertical line for the t_start
    plt.vlines(t_start, 0, max_y_lim_1, colors='k',linewidth = 3)
    
    # Axis format and names
    ax1.set_ylabel(" / ".join((compartment_names[v] for v in plot_left_axis)), fontsize=text_size)
    if ax2 is not None:
        ax2.set_ylabel(compartment_names[plot_right_axis[0]])
    
    # Axis ticks
    ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)])
    ax1.xaxis.set_ticklabels(
        [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)],
        rotation=0,
        fontsize=22)
    for tick in ax1.xaxis.get_major_ticks():
        #tick.tick1line.set_markersize(0)
        #tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('left')
    ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
    ax1.tick_params(axis='x', length=5, width=2)
    
    # Policy axis span 0 - 1
    #policy_ax.yaxis.set_ticks(np.arange(0, 1.001, 0.1))
    policy_ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelbottom=False,
        labelright=False)  # labels along the bottom edge are off
    
    actions_ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off
    actions_ax.spines['top'].set_visible(False)
    actions_ax.spines['bottom'].set_visible(False)
    actions_ax.spines['left'].set_visible(False)
    actions_ax.spines['right'].set_visible(False)

    # if 321 <= T:
    #     # line to separate years
    #     actions_ax.axvline(321, 0, 1, color='k', alpha=0.3)
    if 140 <= T:
        actions_ax.annotate('2020',
                            xy=(140, 0),
                            xycoords='data',
                            color='k',
                            annotation_clip=True,
                            fontsize=text_size - 2)
    if 425 <= T:
        actions_ax.annotate('2021',
                            xy=(425, 0),
                            xycoords='data',
                            color='k',
                            annotation_clip=True,
                            fontsize=text_size - 2)
    
    # Order of layers
    ax1.set_zorder(policy_ax.get_zorder() + 10)  # put ax in front of policy_ax
    ax1.patch.set_visible(False)  # hide the 'canvas'
    if ax2 is not None:
        ax2.set_zorder(policy_ax.get_zorder() + 5)  # put ax in front of policy_ax
        ax2.patch.set_visible(False)  # hide the 'canvas'
    
    # Plot margins
    ax1.margins(0)
    actions_ax.margins(0)
    if ax2 is not None:
        ax2.margins(0)
    policy_ax.margins(0.)
    
    # Plot Grid
    #ax1.grid(True, which='both', color='grey', alpha=0.1, linewidth=0.5, zorder=0)
    
    # fig.delaxes(ax1[1, 2])
    if plot_legend:
        handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels() if ax2 is not None else ([], [])
        handles_action_ax, labels_action_ax = actions_ax.get_legend_handles_labels()
        handles_policy_ax, labels_policy_ax = policy_ax.get_legend_handles_labels()
        plotted_labels = [pl.get_label() for pl in plotted_lines]
        if 'ToIHT' in plot_left_axis or True:
            fig_legend = ax1.legend(
                plotted_lines + handles_policy_ax + handles_action_ax,
                plotted_labels + labels_policy_ax + labels_action_ax,
                loc='upper right',
                fontsize=text_size + 2,
                #bbox_to_anchor=(0.90, 0.9),
                prop={'size': text_size},
                framealpha=1)
        elif 'IH' in plot_left_axis:
            fig_legend = ax1.legend(
                handles_ax1,
                labels_ax1,
                loc='upper right',
                fontsize=text_size + 2,
                #bbox_to_anchor=(0.90, 0.9),
                prop={'size': text_size},
                framealpha=1)
        fig_legend.set_zorder(4)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plots_left_right = plot_left_axis + plot_right_axis
    plot_filename = plots_path / f'scratch_{instance_name}_{"".join(plots_left_right)}.pdf'
    plt.savefig(plot_filename)
    if show:
        plt.show()
    plt.close()
    return plot_filename

def stack_plot(instance_name,
                         instance,
                         policy,
                         profiles,
                         profile_labels,
                         real_hosp,
                         plot_left_axis=['IH'],
                         plot_right_axis=[],
                         scale_plot=False,
                         align_axes=True,
                         show=True,
                         plot_triggers=False,
                         plot_trigger_annotations=False,
                         plot_legend=False,
                         y_lim=None,
                         n_replicas=300,
                         config=None,
                         hosp_beds_list=None,
                         real_new_admission=None,
                         real_hosp_or_icu=None,
                         bed_scale=1,
                         is_representative_path=False,
                         t_start = -1,
                         central_path_id=0,
                         cap_path_id=0,
                         **kwargs):
    '''
    Plots a list of profiles in the same figure. Each profile corresponds
    to a stochastic replica for the given instance.

    Args:
        profiles (list of dict): a list of dictionaries that contain epi vars profiles
        profile_labels (list of str): name of each profile
        plot_only (list of str): list of variable names to be plot
    '''
    plt.rcParams["font.size"] = "18"
    T = kwargs['T']
    if "add_tiers" in kwargs.keys():
        add_tiers = kwargs["add_tiers"]
    cal = instance.cal
    population = instance.N.sum()
    interventions = kwargs['interventions']
    policy_params = kwargs['policy_params']
    if hosp_beds_list is None:
        hosp_beds_list = [instance.hosp_beds]
    hosp_beds = hosp_beds_list[0]
    
    lb_band = 5
    ub_band = 95
    
    text_size = 28
    fig, (ax1, actions_ax) = plt.subplots(2, 1, figsize=(17, 9), gridspec_kw={'height_ratios': [10, 1.1]})
    # Main axis
    # ax1.set_xlabel('Time')
    ax2 = None
    # Policy axis
    policy_ax = ax1.twinx()
    #policy_ax.set_ylabel('Social Distance')
    # If there are plot to be on the right axis, move policy_ax
    # Second, show the right spine.
    if len(plot_right_axis) > 0:
        # Create second axis
        ax2 = ax1.twinx()
        # Fix policy axis
        policy_ax.spines["right"].set_position(("axes", 1.1))
        make_patch_spines_invisible(policy_ax)
        policy_ax.spines["right"].set_visible(True)
    
    # Start plots
    max_y_lim_1 = population if 'S' in plot_left_axis or 'R' in plot_left_axis else 0
    max_y_lim_2 = population if 'S' in plot_right_axis or 'R' in plot_right_axis else 0
    plotted_lines = []
    
    # Add IHT field
    if 'ICU' in profiles[0].keys():
        for p in profiles:
            p['IHT'] = p['IH'] + p['ICU']
    
    # Transform data of interest
    states_to_plot = plot_left_axis + plot_right_axis
    last_day_hosp_data = len(real_hosp) - 1
    lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
    ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
    states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot}
    states_ts['z'] = np.vstack(list(p['z'][:T] for p in profiles))
    states_ts['tier_history'] = np.vstack(list(p['tier_history'][:T] for p in profiles))
    
    if states_to_plot[0] == 'IH':
        states_to_plot_temp = ['ToIHT']
        states_ts_temp = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot_temp}
    else:
        states_to_plot_temp = ['IH']
        states_ts_temp = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in states_to_plot_temp}
        
        
    central_path = 0
    representative_path_id = 0
    print("Printed seed is: ", profiles[0]["seed"])

    if is_representative_path == False:
        central_path = central_path_id
        mean_st = {v: states_ts[v][central_path] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
    else:
        representative_path_id = find_central_path(instance.city, states_to_plot_temp, states_ts_temp, real_hosp, real_hosp_or_icu, real_new_admission)
        mean_st = {v: states_ts[v][representative_path_id] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
        central_path = representative_path_id
        cap_path_id = representative_path_id

    all_st = {v: states_ts[v][:] if v not in ['z', 'tier_history'] else states_ts[v] for v in states_ts}
    min_st = {
        v: np.percentile(states_ts[v], q=lb_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    max_st = {
        v: np.percentile(states_ts[v], q=ub_band, axis=0) if v not in ['z', 'tier_history'] else states_ts[v]
        for v in states_ts
    }
    # People that arrive above capacity
    # np.mean(np.sum(states_ts['IYIH']*(states_ts['IH']>=3239) , 1))
    new_profiles = [mean_st, min_st, max_st]
    
    # Stats
    all_states = ['S', 'E', 'IH', 'IA', 'IY', 'R', 'D']
    if 'ICU' in profiles[0].keys():
        all_states.append('ICU')
        all_states.append('IHT')
        all_states.append('ToICU')
    all_states_ts = {v: np.vstack(list(np.sum(p[v], axis=(1, 2))[:T] for p in profiles)) for v in all_states}
    #assert len(all_states_ts['IH']) >= n_replicas
    for v in all_states_ts:
        all_states_ts[v] = all_states_ts[v][:n_replicas]
    #assert len(all_states_ts['IH']) == n_replicas
        

       
    # Hospitalizations Report
    # Probabilities of reaching x% of the capacity
    prob50 = np.sum(np.any(all_states_ts['IH'] >= 0.5 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob60 = np.sum(np.any(all_states_ts['IH'] >= 0.6 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob70 = np.sum(np.any(all_states_ts['IH'] >= 0.7 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob80 = np.sum(np.any(all_states_ts['IH'] >= 0.8 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob90 = np.sum(np.any(all_states_ts['IH'] >= 0.9 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob100 = np.sum(np.any(all_states_ts['IH'] >= 1 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    prob110 = np.sum(np.any(all_states_ts['IH'] >= 1.1 * hosp_beds, axis=1)) / len(all_states_ts['IH'])
    n_replicas_used = len(all_states_ts['IH'])
    print(f"{'P 50':10s}{'P 60':10s}{'P 70':10s}{'P 80':10s}{'P 90':10s}{'P 100':10s}{'P 110':10s}{'Scenarios':10s}")
    print(
        f"{prob50:<10.4f}{prob60:<10.4f}{prob70:<10.4f}{prob80:<10.4f}{prob90:<10.4f}{prob100:<10.4f}{prob110:<10.4f}{n_replicas_used}"
    )
    # Min, Med, Max at the peak
    print('Hospitalization Peaks')
    peak_days = np.argmax(all_states_ts['IH'], axis=1)
    peak_vals = np.take_along_axis(all_states_ts['IH'], peak_days[:, None], axis=1)
    print(f'{"Percentile (%)":<15s} {"Peak IH":<15s}  {"Date":15}')
    for q in [0, 5, 10, 50, 90, 100]:
        peak_day_percentile = int(np.percentile(peak_days, q))
        peak_percentile = np.percentile(peak_vals, q)
        print(f'{q:<15} {peak_percentile:<15.0f}  {str(cal.calendar[peak_day_percentile])}')
    
    # Deaths
    all_states_ts_ind = {
        v: np.array(list(p[v][:T, :, :] for p in profiles)) for v in all_states
        }
    
    #assert len(all_states_ts_ind['IH']) >= n_replicas
    for v in all_states_ts:
        all_states_ts_ind[v] = all_states_ts_ind[v][:n_replicas]
    #assert len(all_states_ts_ind['IH']) == n_replicas
    # Deaths data
    avg_deaths_by_group = np.round(np.mean(all_states_ts_ind['D'][:, -1, :, :], axis=0).reshape((10, 1)), 0)
    Median_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), 50))
    CI5_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), lb_band))
    CI95_deaths = np.round(np.percentile(np.sum(all_states_ts_ind['D'][:, -1, :, :], axis=(1, 2)), ub_band))
    print('Deaths End Horizon')
    print(f'Point forecast {all_states_ts["D"][0][-1]}')
    print(f'Mean {avg_deaths_by_group.sum()} Median:{Median_deaths} CI_5_95:[{CI5_deaths}-{CI95_deaths}]')
    print('Fraction by Age and Risk Group (1-5, L-H)')
    print(100 * avg_deaths_by_group.reshape(5, 2) / avg_deaths_by_group.sum())
    R_mean = np.mean(all_states_ts['R'][:, -1] / population)
    print(f'R End Horizon {R_mean}')
    # Policy
    lockdown_threshold = policy.lockdown_thresholds[0]
    # fdmi = policy_params['first_day_month_index']
    # policy = {(m, y): lockdown_threshold[fdmi[m, y]] for (m, y) in fdmi if fdmi[m, y] < T}
    # print('Lockdown Threshold:')
    # print(policy)
    central_path = central_path_id
    hide = 1
    l_style = l_styles['sim']
    for v in plot_left_axis:
        max_y_lim_1 = np.maximum(max_y_lim_1, np.max(max_st[v]))
        label_v = compartment_names[v]
        if v != 'IYIHa':
            v_a = ax1.plot(mean_st[v].T * bed_scale, c=colors[v], linestyle=l_style, linewidth=2, label=label_v, alpha=1 * hide, zorder = 50)
            plotted_lines.append(v_a[0])
            v_aa = ax1.plot(all_st[v].T * bed_scale, c=light_colors[v], linestyle=l_style, linewidth=1, label=label_v, alpha=0.8 * hide)
            plotted_lines.append(v_aa[0])
        #if central_path != 0:
        #    ax1.fill_between(range(len(max_st[v])),
        #                     max_st[v],
        #                     min_st[v],
        #                     color=colors[v],
        #                     linestyle=l_style,
        #                     facecolor="none",
        #                     linewidth=0.0,
        #                     alpha=0.5 * hide)
        if v == 'IH' or v == 'ICU' or v == 'IHT':
            real_h_plot = ax1.scatter(range(len(real_hosp_or_icu)), real_hosp_or_icu, color='maroon', label='Actual hospitalizations',zorder=100,s=15)
            max_y_lim_1 = np.maximum(roundup(np.max(hosp_beds_list), 100), max_y_lim_1)
            try:
                if v == 'IH' or v == 'IHT':
                    ax1.plot(profiles[0]['capacity'][:T], color='k', linestyle='-', linewidth=3)
                else:
                    for hosp_beds_lines in hosp_beds_list:
                        ax1.hlines(hosp_beds_lines, 0, T, color='k', linestyle='-', linewidth=3)
            except:
                for hosp_beds_lines in hosp_beds_list:
                    ax1.hlines(hosp_beds_lines, 0, T, color='k', linestyle='-', linewidth=3)
            xpos = 30  #440  #200  # 440
            if plot_trigger_annotations:
                ax1.annotate('Hospital capacity', (xpos, hosp_beds + 150),
                             xycoords='data',
                             color=colors[v],
                             annotation_clip=True,
                             fontsize=text_size + 2)  #
            if plot_triggers:
                ax1.hlines(policy_params['hosp_beds'] * 0.6, 0, T, 'b', '-', linewidth=3)
                if plot_trigger_annotations:
                    ax1.annotate('Safety threshold', (xpos, policy_params['hosp_level_release'] - 250),
                                 xycoords='data',
                                 color='b',
                                 annotation_clip=True,
                                 fontsize=text_size + 2)
        if v == 'ToIHT' or v == 'ToICU':
            if v == 'ToIHT':
                if real_new_admission is not None:
                    real_h_plot = ax1.scatter(range(len(real_new_admission)), real_new_admission, color='maroon', label='New hospital admission',zorder=100,s=15)
            if plot_triggers:
                #if central_path > 0:
                #    IYIH_mov_ave = []
                #    for t in range(T):
                #        IYIH_mov_ave.append(np.mean(mean_st[v][np.maximum(0, t - 7):t]))
                #    v_avg = ax1.plot(IYIH_mov_ave, c='black', linestyle=l_style, label=f'Moving Avg. {label_v}')
                # plotted_lines.append(v_avg[0])
                for tier_ix, tier in enumerate(policy.tiers):
                    ax1.plot([policy.lockdown_thresholds[tier_ix][0]]*T, color=tier['color'], linewidth=5)
                    xpos = np.minimum(405, int(T * 0.65))  #180  #405
                    xytext = (xpos, lockdown_threshold[xpos] - 20)
                    if plot_trigger_annotations:
                        ax1.annotate('Lock-down threshold',
                                     xy=(120, lockdown_threshold[120]),
                                     xytext=xytext,
                                     xycoords='data',
                                     textcoords='data',
                                     color='b',
                                     annotation_clip=True,
                                     fontsize=text_size + 2)
            if "plot_ACS_triggers" in kwargs.keys():
                if kwargs["plot_ACS_triggers"]:
                    ax1.plot([policy.acs_thrs]*T, color='k', linewidth=5)
    for v in plot_right_axis:
        max_y_lim_2 = np.maximum(max_y_lim_2, np.max(max_st[v]))
        label_v = compartment_names[v]
        v_a = ax2.plot(mean_st[v].T, c=colors[v], linestyle=l_style, label=label_v)
        plotted_lines.append(v_a[0])
        ax2.fill_between(range(T), min_st[v], max_st[v], color=colors[v], linestyle=l_style, alpha=0.5)
        if v == 'IH':
            max_y_lim_2 = np.maximum(roundup(hosp_beds, 100), max_y_lim_2)
            ax2.hlines(hosp_beds, 0, T, color='r', linestyle='--', label='N. of beds')
            if plot_triggers:
                ax2.hlines(policy_params['hosp_level_release'], 0, T, 'b', '--')
                ax2.annotate('Trigger - Current hospitalizations ',
                             (0.05, 0.78 * policy_params['hosp_level_release'] / max_y_lim_1),
                             xycoords='axes fraction',
                             color='b',
                             annotation_clip=True)
        if v == 'ToIHT':
            if plot_triggers:
                ax2.plot(lockdown_threshold[:T], 'b-')
                xytext = (160, lockdown_threshold[160] - 15)
                ax2.annotate('Trigger - Avg. Daily Hospitalization',
                             xy=(120, lockdown_threshold[120]),
                             xytext=xytext,
                             xycoords='data',
                             textcoords='data',
                             color='b',
                             annotation_clip=True)
                # ax2.annotate('                                        ',
                #              xy=(85, lockdown_threshold[85]),
                #              xytext=xytext,
                #              xycoords='data',
                #              textcoords='data',
                #              arrowprops={'arrowstyle': '-|>'},
                #              color='b',
                #              annotation_clip=True)
    
    # Plotting the policy
    # Plot school closure and cocooning
    tiers = policy.tiers
    z_ts = profiles[central_path]['z'][:T]
    tier_h = profiles[central_path]['tier_history'][:T]
    print('seed was', profiles[central_path]['seed'])
    sc_co = [interventions[k].school_closure for k in z_ts]
    unique_policies = set(sc_co)
    sd_lvl = [interventions[k].social_distance for k in z_ts]
    sd_levels = [tier['transmission_reduction'] for tier in tiers] + [0, 0.95] + sd_lvl
    unique_sd_policies = list(set(sd_levels))
    unique_sd_policies.sort()
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
    
    interval_color = {0: 'orange', 1: 'purple', 0.5: 'green'}
    interval_labels = {0: 'Schools Open', 1: 'Schools Closed', 0.5: 'Schools P. Open'}
    interval_alpha = {0: 0.3, 1: 0.3, 0.5: 0.3}
    for u in unique_policies:
        u_color = interval_color[u]
        u_label = interval_labels[u]
        
        actions_ax.fill_between(
            range(len(z_ts) + 1),
            0,
            1,
            where=intervals[u],
            color='white',  #u_color,
            alpha=0,  #interval_alpha[u],
            label=u_label,
            linewidth=0,
            hatch = '/',
            step='pre')
        

    # for kv in interval_labels:
    #     kv_label = interval_labels[kv]
    #     kv_color = interval_color[kv]
    #     kv_alpha = interval_alpha[kv]
    #     actions_ax.fill_between(range(len(z_ts) + 1),
    #                             0,
    #                             0.0001,
    #                             color=kv_color,
    #                             alpha=kv_alpha,
    #                             label=kv_label,
    #                             linewidth=0,
    #                             step='pre')
    
    sd_labels = {
        0: '',
        0.95: 'Initial lock-down',
    }
    sd_labels.update({tier['transmission_reduction']: tier['name'] for tier in tiers})
    tier_by_tr = {tier['transmission_reduction']: tier for tier in tiers}
    tier_by_tr[0.746873309820472] = {
        "name": 'Ini Lockdown',
        "transmission_reduction": 0.95,
        "cocooning": 0.95,
        "school_closure": 1,
        "min_enforcing_time": 0,
        "daily_cost": 0,
        "color": 'darkgrey'
    }
        
    if "add_tiers" in kwargs.keys():
        for add_t in add_tiers.keys():
            tier_by_tr[add_t] = {"color": add_tiers[add_t],
                                 "name": "added stage"}
    for u in unique_sd_policies:
        try:
            if u in tier_by_tr.keys():
                u_color = tier_by_tr[u]['color']
                u_label = f'{tier_by_tr[u]["name"]}' if u > 0 else ""
            else:
                u_color,u_label = colorDecide(u,tier_by_tr)
            u_alpha1 = 0.6
            fill_1 = intervals_sd[u].copy()
            fill_2 = intervals_sd[u].copy()
            for i in range(len(intervals_sd[u])):
                if 'history_white' in kwargs.keys() and kwargs['history_white']:
                    if i <= t_start:
                        fill_2[i] = False
                    fill_1[i] = False
                else:
                    if i <= t_start:
                        fill_2[i] = False
                    else:
                        fill_1[i] = False
                    
            policy_ax.fill_between(range(len(z_ts) + 1),
                                   0,
                                   1,
                                   where=fill_1,
                                   color=u_color,
                                   alpha=u_alpha1,
                                   label=u_label,
                                   linewidth=0.0,
                                   step='pre')
            # policy_ax.fill_between(range(len(z_ts) + 1),
            #                        0,
            #                        1,
            #                        where=fill_2,
            #                        color=u_color,
            #                        alpha=u_alpha,
            #                        label=u_label,
            #                        linewidth=0.0,
            #                        step='pre')
        except Exception:
            print(f'WARNING: TR value {u} was not plotted')
            
    # Plot social distance
    social_distance = [interventions[k].social_distance for k in z_ts]
    #policy_ax.plot(social_distance, c='k', alpha=0.6 * hide)  # marker='_', linestyle='None',
    hsd = np.sum(np.array(social_distance[:T]) >= 0.78)
    print(f'HIGH SOCIAL DISTANCE')
    print(f'Point Forecast: {hsd}')
    hsd_list = np.array(
        [np.sum(np.array([interventions[k].social_distance for k in z_ts]) >= 0.78) for z_ts in states_ts['z']])
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
    
    # START PLOT STYLING
    # Axis limits
    if align_axes:
        max_y_lim_1 = np.maximum(max_y_lim_1, max_y_lim_2)
        max_y_lim_2 = max_y_lim_1
    if y_lim is not None:
        max_y_lim_1 = y_lim
    else:
        max_y_lim_1 = roundup(max_y_lim_1, 100 if 'ToIHT' in plot_left_axis else 1000)
    ax1.set_ylim(0, max_y_lim_1)
    policy_ax.set_ylim(0, 1)

    
    # plot the stacked part of the stage proportion
    ax3 = ax1.twinx()
    ax3.set_ylim(0, max_y_lim_1)
        
    data = states_ts['tier_history'].T
    
    tierColor = {}
    for tierInd in range(len(policy.tiers)):
        tierColor[tierInd] = (np.sum(data[(t_start+1):T,:] == tierInd, axis = 1)/len(data[0]))*max_y_lim_1    
  
#     #r = range(len(tier1))
    r = range((t_start+1), T-1)
    bottomTier = 0
    for tierInd in range(len(policy.tiers)):
        ax3.bar(r, tierColor[tierInd], color = policy.tiers[tierInd]['color'], bottom = bottomTier, label = 'tier{}'.format(tierInd), width = 1, alpha = 0.6, linewidth = 0)
        bottomTier += np.array(tierColor[tierInd])
    # ax3.bar(r, tier2, color = 'blue', bottom = np.array(tier1), label = 'tier2', width = 1, alpha = 0.6, linewidth = 0)
    # ax3.bar(r, tier3, color = 'yellow', bottom = np.array(tier1) + np.array(tier2), label = 'tier3', width = 1, alpha = 0.6, linewidth = 0)
    # ax3.bar(r, tier4, color = 'orange', bottom = np.array(tier1) + np.array(tier2) + np.array(tier3), label = 'tier4', width = 1, alpha = 0.6, linewidth = 0)
    # ax3.bar(r, tier5, color = 'red', bottom = np.array(tier1) + np.array(tier2) + np.array(tier3) + np.array(tier4), label = 'tier5', width = 1, alpha = 0.6, linewidth = 0)
    ax3.set_yticks([])
    
    if ax2 is not None:
        ax2.set_ylim(0, roundup(max_y_lim_2, 1000))
    
    # plot a vertical line for the t_start
    plt.vlines(t_start, 0, max_y_lim_1, colors='k',linewidth = 3)
    
    # Axis format and names
    ax1.set_ylabel(" / ".join((compartment_names[v] for v in plot_left_axis)), fontsize=text_size)
    if ax2 is not None:
        ax2.set_ylabel(compartment_names[plot_right_axis[0]])
    
    # Axis ticks
    ax1.xaxis.set_ticks([t for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)])
    ax1.xaxis.set_ticklabels(
        [f' {py_cal.month_abbr[d.month]} ' for t, d in enumerate(cal.calendar) if (d.day == 1 and t < T)],
        rotation=0,
        fontsize=22)
    for tick in ax1.xaxis.get_major_ticks():
        #tick.tick1line.set_markersize(0)
        #tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('left')
    ax1.tick_params(axis='y', labelsize=text_size, length=5, width=2)
    ax1.tick_params(axis='x', length=5, width=2)
    
    # Policy axis span 0 - 1
    #policy_ax.yaxis.set_ticks(np.arange(0, 1.001, 0.1))
    policy_ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelbottom=False,
        labelright=False)  # labels along the bottom edge are off
    
    actions_ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off
    actions_ax.spines['top'].set_visible(False)
    actions_ax.spines['bottom'].set_visible(False)
    actions_ax.spines['left'].set_visible(False)
    actions_ax.spines['right'].set_visible(False)
    # if 321 <= T:
    #     # line to separate years
    #     actions_ax.axvline(321, 0, 1, color='k', alpha=0.3)
    if 140 <= T:
        actions_ax.annotate('2020',
                            xy=(140, 0),
                            xycoords='data',
                            color='k',
                            annotation_clip=True,
                            fontsize=text_size - 2)
    if 425 <= T:
        actions_ax.annotate('2021',
                            xy=(425, 0),
                            xycoords='data',
                            color='k',
                            annotation_clip=True,
                            fontsize=text_size - 2)
    
    # Order of layers
    ax1.set_zorder(policy_ax.get_zorder() + 10)  # put ax in front of policy_ax
    ax1.patch.set_visible(False)  # hide the 'canvas'
    if ax2 is not None:
        ax2.set_zorder(policy_ax.get_zorder() + 5)  # put ax in front of policy_ax
        ax2.patch.set_visible(False)  # hide the 'canvas'
    
    # Plot margins
    ax1.margins(0)
    actions_ax.margins(0)
    if ax2 is not None:
        ax2.margins(0)
    policy_ax.margins(0.)
    
    # Plot Grid
    #ax1.grid(True, which='both', color='grey', alpha=0.1, linewidth=0.5, zorder=0)
    
    # fig.delaxes(ax1[1, 2])
    if plot_legend:
        handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels() if ax2 is not None else ([], [])
        handles_action_ax, labels_action_ax = actions_ax.get_legend_handles_labels()
        handles_policy_ax, labels_policy_ax = policy_ax.get_legend_handles_labels()
        plotted_labels = [pl.get_label() for pl in plotted_lines]
        if 'ToIHT' in plot_left_axis or True:
            fig_legend = ax1.legend(
                plotted_lines + handles_policy_ax + handles_action_ax,
                plotted_labels + labels_policy_ax + labels_action_ax,
                loc='upper right',
                fontsize=text_size + 2,
                #bbox_to_anchor=(0.90, 0.9),
                prop={'size': text_size},
                framealpha=1)
        elif 'IH' in plot_left_axis:
            fig_legend = ax1.legend(
                handles_ax1,
                labels_ax1,
                loc='upper right',
                fontsize=text_size + 2,
                #bbox_to_anchor=(0.90, 0.9),
                prop={'size': text_size},
                framealpha=1)
        fig_legend.set_zorder(4)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plots_left_right = plot_left_axis + plot_right_axis
    plot_filename = plots_path / f'scratch_{instance_name}_{"".join(plots_left_right)}.pdf'
    plt.savefig(plot_filename)
    if show:
        plt.show()
    plt.close()
    return plot_filename

    
def plot_pareto(cost_record, typePlt):
    # plot the pareto frontier with the cost record
    # take the mean of cost record
    plot_record_x = []
    plot_record_y = []
    for iKey in cost_record.keys():
        # each item corresponds to a candidate
        item = cost_record[iKey]
        cost_record_ij = np.array(item)
        lockdown_cost = np.mean(cost_record_ij[:,0])
        over_cap_cost = np.mean(cost_record_ij[:,1])
        plot_record_x.append(lockdown_cost)
        plot_record_y.append(over_cap_cost)
    if typePlt == 's':
        # plot the scatter plot
        plt.scatter(plot_record_x,plot_record_y)
    elif typePlt == 'l':
        # calculate the pareto frontier and plot the line plot
        n_points = len(plot_record_x)
        xy = np.zeros([n_points,2])
        xy[:,0] = np.array(plot_record_x)
        xy[:,1] = np.array(plot_record_y)
        xy_unique = np.unique(xy,axis = 0)
        xy_pareto = is_pareto_efficient_dumb(xy_unique)
        # plot the dots on the pareto frontier
        plt.scatter(xy_unique[xy_pareto][:,0],xy_unique[xy_pareto][:,1])
        # plot the line on the pareto frontier
        xy_pareto_sort = xy_unique[xy_pareto][xy_unique[xy_pareto][:,0].argsort()]
        plt.plot(xy_pareto_sort[:,0],xy_pareto_sort[:,1])
        plt.show()

def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)