import argparse


def parse_arguments():
    parser = argparse.ArgumentParser("Parameters for calibration")
    parser.add_argument("-nworkers", metavar="N2", default=2, type=int, help="Number of workers.")
    parser.add_argument("-minibatch", metavar="N2", default=1, type=int, help="Minibatch size.")
    parser.add_argument(
        "-max_eval",
        metavar="N2",
        default=100,
        type=int,
        help="Number of parameters to acquire.",
    )
    parser.add_argument(
        "-n_init_thetas",
        metavar="N2",
        default=8,
        type=int,
        help="Number of parameters from LHS.",
    )
    parser.add_argument("-seed_n0", metavar="N2", default=1, type=int, help="Seed No.")
    parser.add_argument(
        "-al_func",
        metavar="N2",
        default="eivar",
        type=str,
        help="Acquisition function.",
    )
    parser.add_argument(
        "-funcname",
        metavar="N2",
        default="unimodal",
        type=str,
        help="Name of the function.",
    )
    args = parser.parse_args()
    return args
