import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen
import pandas as pd
from smt.sampling_methods import LHS
import time

args = parse_arguments()


def create_entry(desobj, fname, method, s, b, w):

    r1 = [
        {
            "MAD": MAD,
            "t": t,
            "rep": s,
            "batch": b,
            "worker": w,
            "method": method,
            "example": fname,
        }
        for t, MAD in enumerate(desobj._info["TViter"])
    ]
    
    r2 = [
        {
            "MAD": MAD,
            "t": t,
            "rep": s,
            "batch": b,
            "worker": w,
            "method": method,
            "example": fname,
        }
        for t, MAD in enumerate(desobj._info["TV"])
    ]
    
    return r1, r2

# # # # #
args.minibatch = 8
args.funcname = "unimodal"
args.seedmin = 0
args.seedmax = 30

# # # # #

workers = args.minibatch + 1
n0 = 15
rep0 = 2
nmesh = 50
rho = 1 / 2
batch = args.minibatch
maxiter = 256
dfl1, dfl2 = [], []

# Inputs to designer
pcset = {"standardize": True, "latent": False}
desset = {"is_exploit": True, "is_explore": True, "nL": 200, "impute_str": "update"}

if __name__ == "__main__":
    design_start = time.time()
    for s in np.arange(args.seedmin, args.seedmax):

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

        base_args = {
            "prior": prior_func,
            "data_test": test_data,
            "max_iter": maxiter,
            "nworkers": workers,
            "batch_size": batch,
            "des_init": {"seed": s, "theta": theta0, "f": f0},
            "alloc_settings": {
                "use_Ki": True,
                "rho": rho,
                "theta": None,
                "a0": None,
                "gen": False,
            },
            "pc_settings": pcset,
            "des_settings": desset,
        }

        methods = ["ivar", "imse"]

        args_list = []
        for method in methods:
            args_ = base_args.copy()
            args_["alloc_settings"] = args_["alloc_settings"].copy()
            args_["alloc_settings"]["method"] = method
            args_list.append(args_)

        al_ivar = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="seivar",
            args=args_list[0],
        )

        save_output(al_ivar, cls_func.data_name, "ivar", workers, batch, s)
        e1, e2 = create_entry(al_ivar, args.funcname, "ivar", s, batch, workers)
        dfl1.extend(e1)
        dfl2.extend(e2)

        al_imse = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="simse",
            args=args_list[1],
        )

        save_output(al_imse, cls_func.data_name, "imse", workers, batch, s)
        e1, e2 = create_entry(al_imse, args.funcname, "imse", s, batch, workers)
        dfl1.extend(e1)
        dfl2.extend(e2)

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
                "pc_settings": pcset,
                "des_settings": {"is_exploit": True, "is_explore": True},
            },
        )

        save_output(al_unif, cls_func.data_name, "unif", workers, batch, s)
        e1, e2 = create_entry(al_unif, args.funcname, "unif", s, batch, workers)
        dfl1.extend(e1)
        dfl2.extend(e2)

        al_var = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="var",
            args=args_list[0],
        )

        save_output(al_var, cls_func.data_name, "var", workers, batch, s)
        e1, e2 = create_entry(al_var, args.funcname, "var", s, batch, workers)
        dfl1.extend(e1)
        dfl2.extend(e2)

    design_end = time.time()
    print("Elapsed time: " + str(round(design_end - design_start, 2)))


from summary import lineplot

df = pd.DataFrame(dfl2)
lineplot(df, examples=[args.funcname], batches=[batch])

df = pd.DataFrame(dfl1)
lineplot(df, examples=[args.funcname], batches=[batch])

