import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from utilities import test_data_gen
from smt.sampling_methods import LHS
from joblib import Parallel, delayed
import os
import time

args = parse_arguments()

# # # # #
args.funcname = "banana"
# # # # #


funcname_to_id = {"banana": 1, "bimodal": 2, "unimodal": 3}

n0 = 15
rep0 = 2
nmesh = 50
rho = 1 / 2
maxiter = 256
total_reps = 30

if __name__ == "__main__":
    design_start = time.time()
    print("Running function: " + args.funcname)

    cd = os.getcwd()
    id_ex = funcname_to_id.get(args.funcname, None)
    dp = str(id_ex) + "_" + args.funcname + "_" + "explore"
    if not os.path.exists(dp):
        os.makedirs(dp)
    fp = os.path.join(cd, dp)

    for batch in [8, 16, 32, 64]:
        workers = batch + 1

        sd = os.path.join(fp, str(batch))
        if not os.path.isdir(sd):
            os.makedirs(sd)
        sd = os.path.join(sd, "output/")
        if not os.path.isdir(sd):
            os.makedirs(sd)

        def OneRep(s, batch, workers):
            from test_funcs import bimodal, banana, unimodal

            cls_func = eval(args.funcname)()
            cls_func.realdata(seed=s)

            theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
            test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

            # Set a uniform prior
            prior_func = prior_dist(dist="uniform")(
                a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
            )

            # Set random stream for initial design
            persis_info = {"rand_stream": np.random.default_rng(s)}

            # Initial sample
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
            theta0 = sampling(n0)
            theta0 = np.repeat(theta0, rep0, axis=0)
            f0 = np.zeros((cls_func.d, n0 * rep0))
            for i in range(0, n0 * rep0):
                f0[:, i] = cls_func.sim_f(theta0[i, :], persis_info=persis_info)

            al_ivar = designer(
                data_cls=cls_func,
                method="p_sto_bseq",
                acquisition="seivar",
                args={
                    "prior": prior_func,
                    "data_test": test_data,
                    "max_iter": maxiter,
                    "nworkers": workers,
                    "batch_size": batch,
                    "des_init": {"seed": s, "theta": theta0, "f": f0},
                    "alloc_settings": {
                        "method": "ivar",
                        "use_Ki": True,
                        "rho": rho,
                        "theta": None,
                        "a0": None,
                        "gen": False,
                    },
                    "pc_settings": {"standardize": True, "latent": False},
                    "des_settings": {
                        "is_exploit": True,
                        "is_explore": True,
                        "nL": 200,
                        "impute_str": "update",
                    },
                },
            )

            save_output(al_ivar, cls_func.data_name, "ivar", workers, batch, s, sd)

            al_imse = designer(
                data_cls=cls_func,
                method="p_sto_bseq",
                acquisition="simse",
                args={
                    "prior": prior_func,
                    "data_test": test_data,
                    "max_iter": maxiter,
                    "nworkers": workers,
                    "batch_size": batch,
                    "des_init": {"seed": s, "theta": theta0, "f": f0},
                    "alloc_settings": {
                        "method": "imse",
                        "use_Ki": True,
                        "rho": rho,
                        "theta": None,
                        "a0": None,
                        "gen": False,
                    },
                    "pc_settings": {"standardize": True, "latent": False},
                    "des_settings": {
                        "is_exploit": True,
                        "is_explore": True,
                        "nL": 200,
                        "impute_str": "update",
                    },
                },
            )

            save_output(al_imse, cls_func.data_name, "imse", workers, batch, s, sd)

            rholhs = 1 / 4
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
            theta_lhs = sampling(int(maxiter * rholhs))

            al_unif = designer(
                data_cls=cls_func,
                method="p_sto_bseq",
                acquisition=None,
                args={
                    "prior": prior_func,
                    "data_test": test_data,
                    "max_iter": maxiter,
                    "nworkers": workers,
                    "batch_size": batch,
                    "des_init": {"seed": s, "theta": theta0, "f": f0},
                    "des_add": {"theta": theta_lhs},
                    "alloc_settings": {
                        "use_Ki": True,
                        "rho": rholhs,
                        "theta": None,
                        "a0": None,
                        "gen": False,
                    },
                    "pc_settings": {"standardize": True, "latent": False},
                    "des_settings": {"is_exploit": True, "is_explore": True},
                },
            )

            save_output(al_unif, cls_func.data_name, "unif", workers, batch, s, sd)

            al_var = designer(
                data_cls=cls_func,
                method="p_sto_bseq",
                acquisition="var",
                args={
                    "prior": prior_func,
                    "data_test": test_data,
                    "max_iter": maxiter,
                    "nworkers": workers,
                    "batch_size": batch,
                    "des_init": {"seed": s, "theta": theta0, "f": f0},
                    "alloc_settings": {
                        "method": "ivar",
                        "use_Ki": True,
                        "rho": rho,
                        "theta": None,
                        "a0": None,
                        "gen": False,
                    },
                    "pc_settings": {"standardize": True, "latent": False},
                    "des_settings": {
                        "is_exploit": True,
                        "is_explore": True,
                        "nL": 200,
                        "impute_str": "update",
                    },
                },
            )

            save_output(al_var, cls_func.data_name, "var", workers, batch, s, sd)

        Parallel(n_jobs=10)(
            delayed(OneRep)(rep_no, batch, workers) for rep_no in range(total_reps)
        )
    design_end = time.time()
    print("Elapsed time: " + str(round(design_end - design_start, 2)))
