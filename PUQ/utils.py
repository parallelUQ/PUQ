import argparse
import dill as pickle
import os

def parse_arguments():
    parser = argparse.ArgumentParser("Parameters for calibration")
    parser.add_argument("-nworkers",
                        metavar='N2',
                        default=2,
                        type=int,
                        help='Number of workers.')
    parser.add_argument("-minibatch",
                        metavar='N2',
                        default=1,
                        type=int,
                        help='Minibatch size.')
    parser.add_argument("-max_eval",
                        metavar='N2',
                        default=100,
                        type=int,
                        help='Number of parameters to acquire.')
    parser.add_argument("-seedmin",
                        metavar='N2',
                        default=0,
                        type=int,
                        help='Initial seed.')
    parser.add_argument("-seedmax",
                        metavar='N2',
                        default=30,
                        type=int,
                        help='Final seed.')
    parser.add_argument("-seed_n0",
                        metavar='N2',
                        default=1,
                        type=int,
                        help='Seed No.') 
    parser.add_argument("-al_func",
                        metavar='N2',
                        default='eivar',
                        type=str,
                        help='Acquisition function.') 
    args = parser.parse_args()
    return args

def save_output(desing_obj, name, al_func, nworker, minibatch, seedno):
    if not os.path.isdir('output'):
        os.mkdir('output')
        
    design_path = 'output/' + name + '_' + al_func + '_w_' + str(nworker) + '_b_' + str(minibatch) + '_seed_' + str(seedno) + '.pkl'
    with open(design_path, 'wb') as file:
        pickle.dump(desing_obj, file)
    
def read_output(path1, name, al_func, nworker, minibatch, seedno):
    
    design_path = path1 + 'output/' + name + '_' + al_func + '_w_' + str(nworker) + '_b_' + str(minibatch) + '_seed_' + str(seedno) + '.pkl'
    with open(design_path, 'rb') as file:
        design_obj = pickle.load(file) 
        
    return design_obj