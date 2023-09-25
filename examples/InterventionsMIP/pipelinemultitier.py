import pickle
import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
from InterventionsMIP import load_config_file,config_path
#from reporting.plotting import plot_multi_tier_sims, stack_plot
#from reporting.report_pdf import generate_report
#from reporting.output_processors import build_report,build_report_tiers

def read_hosp(file_path, start_date, typeInput="hospitalized"):
    with open(file_path, 'r') as hosp_file:
        df_hosp = pd.read_csv(
            file_path,
            parse_dates=['date'],
            date_parser=pd.to_datetime,
        )
    # if hospitalization data starts before start_date 
    if df_hosp['date'][0] <= start_date:
        df_hosp = df_hosp[df_hosp['date'] >= start_date]
        real_hosp = list(df_hosp[typeInput])
    else:
        real_hosp = [0] * (df_hosp['date'][0] - start_date).days + list(df_hosp[typeInput])
    
    return real_hosp

def multi_tier_pipeline(file_path, instance_name, real_hosp=None, real_admit=None, hosp_beds_list=None, to_email=None):
    
    # Read data
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
    
    # Get only desired profiles
    if real_hosp is None:
        real_hosp = instance.cal.real_hosp
    last_day_hosp_data = len(real_hosp) - 1
    lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
    ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
    profiles = [p for p in profiles]
    n_replicas = len(profiles)
    T = np.minimum(instance.T, instance.T)  #229
    IHD_plot = plot_multi_tier_sims(instance_name,
                                    instance,
                                    best_policy,
                                    profiles, ['sim'] * len(profiles),
                                    real_hosp,
                                    plot_left_axis=['IH'],
                                    plot_right_axis=[],
                                    T=T,
                                    interventions=interventions,
                                    show=True,
                                    align_axes=True,
                                    plot_triggers=False,
                                    plot_trigger_annotations=False,
                                    plot_legend=False,
                                    y_lim=None,
                                    policy_params=best_params,
                                    n_replicas=n_replicas,
                                    config=config,
                                    hosp_beds_list=hosp_beds_list,
                                    real_new_admission=real_admit)
    
    IYIH_plot = plot_multi_tier_sims(instance_name,
                                     instance,
                                     best_policy,
                                     profiles, ['sim'] * len(profiles),
                                     real_hosp,
                                     plot_left_axis=['IYIH'],
                                     plot_right_axis=[],
                                     T=T,
                                     interventions=interventions,
                                     show=True,
                                     align_axes=False,
                                     plot_triggers=True,
                                     plot_trigger_annotations=False,
                                     plot_legend=False,
                                     y_lim=None,
                                     policy_params=best_params,
                                     n_replicas=n_replicas,
                                     config=config,
                                     hosp_beds_list=hosp_beds_list,
                                     real_new_admission=real_admit)    
    build_report(instance_name,
                  instance,
                  best_policy,
                  profiles,
                  IHD_plot,
                  IYIH_plot,
                  to_email=to_email,
                  T=T,
                  hosp_beds=instance.hosp_beds,
                  interventions=interventions,
                  policy_params=best_params,
                  n_replicas=n_replicas,
                  config=config)
    
def icu_pipeline(file_path, instance_name, real_hosp=None, real_admit=None, hosp_beds_list=None, icu_beds_list=None, real_icu=None, 
                 iht_limit=None, icu_limit=None, toiht_limit=None, toicu_limit=None, t_start = -1, to_email=None, is_representative_path_bool=False,
                 central_id_path = 0, cap_id_path = 0):
    
    # Read data
    with open(file_path, 'rb') as outfile:
        read_output = pickle.load(outfile)
    instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
    
    # Get only desired profiles
    if real_hosp is None:
        real_hosp = instance.cal.real_hosp
    last_day_hosp_data = len(real_hosp) - 1
    lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
    ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
    profiles = [p for p in profiles]
    n_replicas = len(profiles)
    T = np.minimum(instance.T, instance.T)  #229
    
    plot_trigger_ToICU = False
    plot_trigger_ToIHT = False
    plot_trigger_ICU = False
    plot_trigger_ToIHT = True
        
    icu_beds_list = [instance.icu]
    
    # plot the IHT comparison
    IHD_plot = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_icu,
                            plot_left_axis=['ICU'],
                            plot_right_axis=[],
                            T=T,
                            interventions=interventions,
                            show=False,
                            align_axes=True,
                            plot_triggers=plot_trigger_ICU,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=icu_limit,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            hosp_beds_list= icu_beds_list,
                            real_new_admission=real_admit,
                            real_hosp_or_icu=real_icu,
                            t_start = t_start,
                            is_representative_path=is_representative_path_bool,
                            central_path_id = central_id_path,
                            cap_path_id = cap_id_path,
                            vertical_fill = not plot_trigger_ICU,
                            history_white = True
                            )
    
    # plot the IHT comparison
    IHD_plot2 = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['IHT'],
                            plot_right_axis=[],
                            T=T,
                            interventions=interventions,
                            show=False,
                            align_axes=True,
                            plot_triggers=False,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=iht_limit,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            hosp_beds_list= hosp_beds_list,
                            real_new_admission=real_admit,
                            real_hosp_or_icu=real_hosp,
                            t_start = t_start,
                            is_representative_path=is_representative_path_bool,
                            central_path_id = central_id_path,
                            cap_path_id = cap_id_path,
                            history_white = True
                            )
    
    # plot the ToICU comparison
    IYIH_plot = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['ToICU'],
                            plot_right_axis=[],
                            T=T,
                            interventions=interventions,
                            show=False,
                            align_axes=True,
                            plot_triggers=plot_trigger_ToICU,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=toicu_limit,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            hosp_beds_list=None,
                            real_new_admission=None,
                            t_start = t_start,
                            is_representative_path=False,
                            central_path_id = central_id_path,
                            cap_path_id = cap_id_path,
                            history_white = True
                            )
    
    IYIH_plot2 = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['ToIHT'],
                            plot_right_axis=[],
                            T=T,
                            interventions=interventions,
                            show=False,
                            align_axes=True,
                            plot_triggers=plot_trigger_ToIHT,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=toiht_limit,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            hosp_beds_list=None,
                            real_new_admission=real_admit,
                            real_hosp_or_icu=real_icu,
                            t_start = t_start,
                            is_representative_path=is_representative_path_bool,
                            central_path_id = central_id_path,
                            cap_path_id = cap_id_path,
                            vertical_fill = not plot_trigger_ToIHT,
                            nday_avg = 7,
                            history_white = True
                            )   
        
    IHT_stacked_plot = stack_plot(instance_name+"_stacked",
                                  instance,
                                  best_policy,
                                  profiles, ['sim'] * len(profiles),
                                  real_hosp,
                                  plot_left_axis=['ICU'],
                                  plot_right_axis=[],
                                  T=T,
                                  interventions=interventions,
                                  show=False,
                                  align_axes=False,
                                  plot_triggers=False, #set false when looking at total hospitalizations
                                  plot_trigger_annotations=False,
                                  plot_legend=False,
                                  y_lim=icu_limit,
                                  policy_params=best_params,
                                  n_replicas=n_replicas,
                                  config=config,
                                  hosp_beds_list=icu_beds_list,
                                  real_new_admission=real_admit,
                                  real_icu_patients=real_icu,
                                  real_hosp_or_icu=real_icu,
                                  period=1,
                                  is_representative_path=False,
                                  t_start = t_start,
                                  central_path_id = central_id_path,
                                  cap_path_id = central_id_path,
                                  history_white = True)
    
    build_report_tiers(instance_name,
                        instance,
                        best_policy,
                        profiles,
                        IHD_plot2,
                        IYIH_plot2,
                        IHD_plot,
                        IHT_stacked_plot,
                        n_replicas = len(profiles),
                        config=config,
                        T=T,
                        hosp_beds=hosp_beds_list[0],
                        icu_beds=icu_beds_list[0],
                        interventions=interventions,
                        policy_params=best_params,
                        stat_start=instance.cal.calendar[t_start+1]
                        )

if __name__ == "__main__":
    # list all .p files from the output folder
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print(dname)
    fileList = os.listdir("output")
    for instance_raw in fileList:
        if ".p" in instance_raw:
            end_history = dt(2020,10,7)
            if "austin" in instance_raw:
                file_path = "instances/austin/austin_real_hosp_updated.csv"
                start_date = dt(2020,2,28)
                real_hosp = read_hosp(file_path, start_date)
                hosp_beds_list = None
                file_path = "instances/austin/austin_hosp_ad_updated.csv"
                hosp_ad = read_hosp(file_path, start_date, "admits")
                file_path = "instances/austin/austin_real_icu_updated.csv"
                real_icu = read_hosp(file_path, start_date)
                iht_limit = 2000
                icu_limit = 500
                toiht_limit = 150
                toicu_limit = 100
                hosp_beds_list = [1500]
                icu_beds_list = None
                t_start = (end_history - start_date).days
                central_id_path = 0
            elif "houston" in instance_raw:
                file_path = "instances/houston/houston_real_hosp_updated.csv"
                start_date = dt(2020,2,19)
                real_hosp = read_hosp(file_path, start_date)
                hosp_beds_list = None
                file_path = "instances/houston/houston_real_icu_updated.csv"
                real_icu = read_hosp(file_path, start_date)
                hosp_ad = None
                if "tiers1" in instance_raw:
                    hosp_beds_list = [4500,9000,13500]
                else:
                    hosp_beds_list = None
                hosp_ad = None
                iht_limit = 6000
                icu_limit = 3000
                toiht_limit = 700
                toicu_limit = 200
                hosp_beds_list = [4500]
                icu_beds_list = [1000]
                t_start = (end_history - start_date).days
                central_id_path = 0
                
            instance_name = instance_raw[:-2]
            path_file = f'output/{instance_name}.p'
            icu_pipeline(path_file, instance_name, real_hosp, hosp_ad, hosp_beds_list, icu_beds_list, real_icu,
                          iht_limit, icu_limit, toiht_limit, toicu_limit, t_start, None, False, central_id_path, central_id_path)
