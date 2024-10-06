import argparse


def parse_arguments():
    parser = argparse.ArgumentParser("Parameters for calibration")
    parser.add_argument(
        "-nworkers", metavar="N2", default=2, type=int, help="Number of workers."
    )
    parser.add_argument(
        "-minibatch", metavar="N2", default=1, type=int, help="Minibatch size."
    )
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
    parser.add_argument(
        "-candsize", metavar="N2", default=100, type=int, help="Candidate size."
    )
    parser.add_argument(
        "-refsize", metavar="N2", default=100, type=int, help="Reference list size."
    )
    parser.add_argument(
        "-believer", metavar="N2", default=0, type=int, help="Kriging believer type."
    )
    args = parser.parse_args()
    return args


def save_output(desing_obj, name, al_func, nworker, minibatch, seedno):
    outputname = (
        "output_" + name + "_" + al_func + "_w_" + str(nworker) + "_b_" + str(minibatch)
    )
    if not os.path.isdir(outputname):
        os.mkdir(outputname)

    design_path = (
        outputname
        + "/"
        + name
        + "_"
        + al_func
        + "_w_"
        + str(nworker)
        + "_b_"
        + str(minibatch)
        + "_seed_"
        + str(seedno)
        + ".pkl"
    )
    with open(design_path, "wb") as file:
        pickle.dump(desing_obj, file)


def read_output(path1, name, al_func, nworker, minibatch, seedno):
    outputname = "output"  # + name + '_' + al_func + '_w_' + str(nworker) + '_b_' + str(minibatch)
    design_path = (
        path1
        + outputname
        + "/"
        + name
        + "_"
        + al_func
        + "_w_"
        + str(nworker)
        + "_b_"
        + str(minibatch)
        + "_seed_"
        + str(seedno)
        + ".pkl"
    )
    with open(design_path, "rb") as file:
        design_obj = pickle.load(file)

    return design_obj

