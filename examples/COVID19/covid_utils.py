import os
import sys
import numpy as np
import time
import argparse
import calendar as py_cal
from pathlib import Path
from collections import defaultdict

plots_path = Path(__file__).parent / 'plots'

profile_log = {}


def roundup(x, l):
    return int(np.ceil(x / l)) * l


def round_closest(x, l):
    return int(np.around(x / l)) * l


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        name = kw.get('log_name', method.__name__)
        call_time = (te - ts)
        if name not in profile_log:
            profile_log[name] = {'calls': 1, 'max_time': call_time, 'avg_time': call_time}
        else:
            avg = profile_log[name]['avg_time']
            calls = profile_log[name]['calls']
            profile_log[name]['avg_time'] = (avg * calls + call_time) / (calls + 1)
            profile_log[name]['calls'] = calls + 1
            profile_log[name]['max_time'] = np.maximum(profile_log[name]['max_time'], call_time)
        
        # else:
        #     print(f'{method.__name__}: {(te - ts): 10.5f} s')
        return result
    
    return timed


def print_profiling_log(logger):
    for (k, v) in profile_log.items():
        logger.info(f'{str(k):15s}: {str(v)}')


def parse_arguments():

    parser = argparse.ArgumentParser("Parameters for trigger optimization")
    parser.add_argument("-city", default='austin',  type=str, help='City to simulate')
    parser.add_argument("-f",
                        metavar='FILENAME',
                        default='setup_data.json',
                        type=str,
                        help='File name where setup data is (stem only).')
    parser.add_argument("-t",
                        metavar='FILENAME',
                        default='tiers.json',
                        type=str,
                        help='File name where tier data is (stem only).')
    parser.add_argument("-tr",
                        metavar='FILENAME',
                        default='transmission.csv',
                        type=str,
                        help='File name where transmission reduction data is (stem only).')
    parser.add_argument("-train_reps",
                        metavar='N1',
                        default=300,
                        type=int,
                        help='Number of replicas when training the policy.')
    parser.add_argument("-test_reps",
                        metavar='N2',
                        default=350,
                        type=int,
                        help='Number of replicas when testing the policy.')
    parser.add_argument("-f_config",
                        metavar='FILENAME',
                        default='trigger_config.json',
                        type=str,
                        help='File name with opt configurations.')
    parser.add_argument("-seed",
                        metavar='FILENAME',
                        default='seeds.p',
                        type=str,
                        help='File name with opt configurations.')
    parser.add_argument("-hos",
                        metavar='FILENAME',
                        default='hospitalizations.csv',
                        type=str,
                        help='File name with real hospitalizations.')
    parser.add_argument("-n_proc", metavar='NP', default=4, type=int, help='Number of processor to use')
    # parser.add_argument("-task", metavar='TASK', default='optimize', type=str, help='Task to be excecuted')
    # parser.add_argument("-n_tiers",
    #                     metavar='TIERS',
    #                     default=5,
    #                     type=int,
    #                     help='Number of tiers to use, options are 2 and 5.')
    parser.add_argument("-agg", action="store_true", help='Whether to use agg for matplotlib')
    parser.add_argument("-plot", action="store_true", help='Whether to use agg for matplotlib')
    parser.add_argument("-machine", default='local', type=str, help='Whether to use agg for matplotlib')
    parser.add_argument("-gt", default=None, help='Given tier levels, in the format of list or list of lists, without space in between')
    parser.add_argument("-gd", default=None, help='Given step date, in the format of year=yyyy,month=mm,day=dd, without space in between')
    parser.add_argument("-field", default='IYIH', type=str, help='The field selected for criterion')
    parser.add_argument("-fo", default='[]', type=str, help='Forced out tiers before first red')
    parser.add_argument("-rl", default=1000, type=int, help='How long is the maximum length of red')
    parser.add_argument("-aftert", default='[0,1,2,3,4]', type=str, help='Tiers after first red')
    parser.add_argument("-pub", default=None, help='Policy upper bound')
    
    args = parser.parse_args()
    
    if args.agg:
        import matplotlib
        matplotlib.use('agg')
        print('Agg in use')
    return args





#scp -r InterventionsMIP dduque@crunch.osl.northwestern.edu:~/scratch/TriggerOptimization/
#scp dduque@crunch.osl.northwestern.edu:~/scratch/TriggerOptimization/InterventionsMIP/output/austin_step_Stage1_0_0_Stage2_0_5_Stage3_5_50_Stage4_30_80_Stage5_50_88_test_300.p ./InterventionsMIP/output/
#scp frontera.tacc.utexas.edu:\$SCRATCH/InterventionsMIP/output/austin*.p ./InterventionsMIP/output/
#InterventionsMIP/output/austin_SC0_CO0.8_BLTrain0.4_BLTest_0.4_step_opt.p  austin_SC0_CO0.95ST_BLTrain0.4_BLTest_0.4_step_opt
# austin_step_Stage1_0_0_Stage2_0_5_Stage3_5_15_Stage4_25_60_Stage5_60_88_test_300
# scp InterventionsMIP/*.py dduque@crunch.osl.northwestern.edu:~/scratch/TriggerOptimization/InterventionsMIP
# scp -r InterventionsMIP/reporting dduque@crunch.osl.northwestern.edu:~/scratch/TriggerOptimization/InterventionsMIP
# scp -r InterventionsMIP/instances dduque@crunch.osl.northwestern.edu:~/scratch/TriggerOptimization/InterventionsMIP
# scp -r InterventionsMIP/config dduque@crunch.osl.northwestern.edu:~/scratch/TriggerOptimization/InterventionsMIP