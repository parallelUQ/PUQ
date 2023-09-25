from InterventionsMIP import project_path, instances_path
import multiprocessing as mp
from threshold_policy import threshold_policy_search
from interventions import Intervension
from epi_params import EpiSetup, ParamDistribution
from utils import parse_arguments
from reporting.plotting import plot_stoch_simulations

from instances import load_instance

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Parse city and get corresponding instance
    instance = load_instance(args.city, setup_file_name=args.f)
    
    # TODO Read command line args for n_proc for better integration with crunch
    n_proc = args.n_proc
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = args.train_reps
    n_replicas_test = args.test_reps
    # Create the pool (Note: pool needs to be created only once to run on a cluster)
    mp_pool = mp.Pool(n_proc) if n_proc > 1 else None
    for sc in [0]:
        for co in [0.95]:
            for base_line_train in [0.4]:
                for base_line_test in [0.4]:
                    for const in ['test']:  #[10 * i for i in range(0, 21)] + [215, 1000]:
                        policy_class = 'step'
                        instance_name = f'local_{instance.city}_SC{sc}_CO{co}_BLTrain{base_line_train}_BLTest_{base_line_test}_{policy_class}_{const}'
                        print('\n============================================')
                        print(instance_name)
                        #TODO: This list should be longe to include all possible transmission reduction values
                        #       that might come in the instance file
                        interventions_train = [
                            Intervension(0, 0, 0, instance.epi, instance.N),
                            Intervension(1, 0, 0, instance.epi, instance.N),
                            Intervension(0, 0, base_line_train, instance.epi, instance.N),
                            Intervension(1, 0, base_line_train, instance.epi, instance.N),
                            Intervension(1, 0, 0.9, instance.epi, instance.N),
                            Intervension(0, co, base_line_train, instance.epi, instance.N),
                            Intervension(1, co, base_line_train, instance.epi, instance.N),
                            Intervension(1, co, 0.9, instance.epi, instance.N),
                            Intervension(1, 0, 0.95, instance.epi, instance.N),
                            Intervension(0, 0, 0.95, instance.epi, instance.N)
                        ]
                        interventions_test = [
                            Intervension(0, 0, 0, instance.epi, instance.N),
                            Intervension(1, 0, 0, instance.epi, instance.N),
                            Intervension(0, 0, base_line_test, instance.epi, instance.N),
                            Intervension(1, 0, base_line_test, instance.epi, instance.N),
                            Intervension(1, 0, 0.9, instance.epi, instance.N),
                            Intervension(0, co, base_line_test, instance.epi, instance.N),
                            Intervension(1, co, base_line_test, instance.epi, instance.N),
                            Intervension(1, co, 0.9, instance.epi, instance.N),
                            Intervension(1, 0, 0.95, instance.epi, instance.N),
                            Intervension(0, 0, 0.95, instance.epi, instance.N)
                        ]
                        sd_levels_train = {'H': 0.9, 'L': base_line_train}
                        sd_levels_test = {'H': 0.9, 'L': base_line_test}
                        best_policy_replicas, policy_params = threshold_policy_search(instance,
                                                                                      interventions_train,
                                                                                      interventions_test,
                                                                                      sd_levels_train,
                                                                                      sd_levels_test,
                                                                                      cocooning=co,
                                                                                      school_closure=sc,
                                                                                      mp_pool=mp_pool,
                                                                                      n_replicas_train=n_replicas_train,
                                                                                      n_replicas_test=n_replicas_test,
                                                                                      instance_name=instance_name,
                                                                                      policy={
                                                                                          'class': policy_class,
                                                                                          'vals': [120, 216, 9]
                                                                                      },
                                                                                      policy_class=policy_class)
                        
                        n_replicas = len(best_policy_replicas)
                        plot_stoch_simulations(
                            instance_name,
                            best_policy_replicas,
                            ['sim'] * n_replicas,
                            plot_left_axis=['IH'],
                            plot_right_axis=[],
                            T=instance.T,  #437,
                            hosp_beds=instance.hosp_beds,
                            population=instance.N.sum(),
                            interventions=interventions_test,
                            calendar=instance.cal,
                            policy_params=policy_params,
                            plot_triggers=True,
                            plot_legend=True,
                            show=True,
                            align_axes=True,
                            n_replicas=5,
                            BL=base_line_test)
