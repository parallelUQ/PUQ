import pickle
import os
print(os.getcwd())
import numpy as np
from reporting.plotting import plot_stoch_simulations
from reporting.report_pdf import generate_report
from reporting.output_processors import build_report


def multi_tier_plotting(instance, tiers):
    pass


BL = 0.7
n_replicas = 300
to_email = None  # 'some_email_address@gmail.com'
for tr in ['opt']:
    instance_name = f'austin_SC0_CO0.95_BLTrain{BL}_BLTest_{BL}_constant_{tr}'
    #instance_name = 'newcap_90_80_95'
    print(instance_name)
    with open(f'./InterventionsMIP/output/{instance_name}.p', 'rb') as outfile:
        read_output = pickle.load(outfile)
    instance_summary, interventions, best_params, best_policy, best_sim, best_params, replicas = read_output
    (epi, T, A, L, N, I0, hosp_beds, lambda_star, cal) = instance_summary
    population = N.sum()
    print_params = [
        'moving_avg_len', 'hosp_rate_threshold_1', 'hosp_rate_threshold_2', 'hosp_rate_threshold_3',
        'hosp_level_release'
    ]
    print([(k, best_params[k]) for k in print_params])
    
    T = np.minimum(229, T)
    IHD_plot = plot_stoch_simulations(instance_name,
                                      replicas, ['sim'] * len(replicas),
                                      plot_left_axis=['IH', 'D'],
                                      plot_right_axis=[],
                                      T=T,
                                      hosp_beds=hosp_beds,
                                      population=population,
                                      interventions=interventions,
                                      show=False,
                                      align_axes=True,
                                      plot_triggers=True,
                                      plot_trigger_annotations=False,
                                      plot_legend=False,
                                      y_lim=1500,
                                      calendar=cal,
                                      policy_params=best_params,
                                      BL=BL,
                                      n_replicas=n_replicas)
    IYIH_plot = plot_stoch_simulations(instance_name,
                                       replicas, ['sim'] * len(replicas),
                                       plot_left_axis=['IYIH'],
                                       plot_right_axis=[],
                                       T=T,
                                       hosp_beds=hosp_beds,
                                       population=population,
                                       interventions=interventions,
                                       show=True,
                                       align_axes=False,
                                       plot_triggers=True,
                                       plot_trigger_annotations=False,
                                       plot_legend=False,
                                       y_lim=200,
                                       calendar=cal,
                                       policy_params=best_params,
                                       BL=BL,
                                       n_replicas=n_replicas)
    
    build_report(instance_name,
                 replicas,
                 IHD_plot,
                 IYIH_plot,
                 to_email=to_email,
                 T=T,
                 hosp_beds=hosp_beds,
                 population=population,
                 interventions=interventions,
                 calendar=cal,
                 policy_params=best_params,
                 BL=BL,
                 n_replicas=n_replicas)
