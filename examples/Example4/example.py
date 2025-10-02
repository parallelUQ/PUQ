import numpy as np
from PUQ.prior import prior_dist
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen, twodpaper
from smt.sampling_methods import LHS
from PUQ.designmethods.sequential_md_stochastic import sequential_design


# # # # #
batch = 8
workers = batch + 1
funcname = "unimodal"
smin = 0
smax = 1

# # # # #


n0, rep0, rho = 15, 2, 1 / 2

nmesh = 50
maxiter = 10


# Inputs to designer
desset = {"is_exploit": True, "is_explore": True, "nL": 200, "impute_str": "update"}

if __name__ == "__main__":

    for s in np.arange(smin, smax):

        cls_func = eval(funcname)()
        cls_func.realdata(seed=s)

        theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
        test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

        # heatmap(cls_func)

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
            "batch_size": batch,
            "alloc_settings": {
                "use_Ki": True,
                "rho": rho,
                "theta": None,
                "a0": None,
                "gen": False,
            },
            "des_settings": desset,
        }

        methods = ["ivar"]

        args_list = []
        for method in methods:
            args_ = base_args.copy()
            args_["alloc_settings"] = args_["alloc_settings"].copy()
            args_["alloc_settings"]["method"] = method
            args_list.append(args_)

        des_obj = sequential_design(cls_func)
        des_obj.build_design(t0=theta0, f0=f0, af="seivar", args=args_list[0])
        
        theta = des_obj["theta0"]
        reps = des_obj["rep0"]

        twodpaper(cls_func, Xpl, Ypl, p_test, theta, reps, thetainit=theta0, name="fig1")

