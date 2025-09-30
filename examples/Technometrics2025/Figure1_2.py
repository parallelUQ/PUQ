import numpy as np
from PUQ.prior import prior_dist
#from PUQ.utils import parse_arguments
from PUQ.design import designer
from test_funcs import sinf
from utilities import test_data_gen_1d, Figure1, Figure2
import matplotlib.pyplot as plt

#args = parse_arguments()

class Args:
    pass

args = Args()

# # # # #
args.minibatch = 100
example = "sinf"
args.seedmin = 13
args.seedmax = 14
# # # # #

datalist = []
n0 = 20
rep0 = 5
batch = args.minibatch
maxiter = 100
worker = args.minibatch + 1
rho = 1 / 2
if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):

        cls_func = eval(example)()
        cls_func.realdata(s)

        theta_test, p_test, f_test = test_data_gen_1d(cls_func, 100)
        test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

        # Set a uniform prior
        prior_func = prior_dist(dist="uniform")(
            a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
        )

        # Set random stream for initial design
        persis_info = {"rand_stream": np.random.default_rng(s)}

        theta = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], n0)[
            :, None
        ]
        theta = np.repeat(theta, rep0, axis=0)
        f = np.zeros((cls_func.d, rep0 * n0))
        for i in range(0, rep0 * n0):
            f[:, i] = cls_func.sim_f(theta[i, :], persis_info=persis_info)

        theta_test, nugs, phat = Figure1(
            f,
            theta,
            cls_func.x,
            cls_func.obsvar,
            cls_func.real_data,
            theta_test,
            f_test,
            p_test,
            cls_func,
        )

        fig, ax = plt.subplots(1, 2, figsize=(15, 3.5))
        fig.subplots_adjust(wspace=0.6)

        al_ivar = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="seivar",
            args={
                "prior": prior_func,
                "data_test": test_data,
                "max_iter": maxiter,
                "nworkers": worker,
                "batch_size": batch,
                "des_init": {"seed": s, "theta": theta, "f": f},
                "alloc_settings": {
                    "method": "ivar",
                    "use_Ki": True,
                    "theta": None,
                    "a0": None,
                    "gen": False,
                    "rho": rho,
                },
                "des_settings": {"is_explore": False, "is_exploit": True},
                "pc_settings": {"standardize": True, "latent": False},
                "trace": 0,
            },
        )

        Figure2(al_ivar, theta_test, nugs, phat, "ivar", ax[0])

        al_imse = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="simse",
            args={
                "prior": prior_func,
                "data_test": test_data,
                "max_iter": maxiter,
                "nworkers": worker,
                "batch_size": batch,
                "des_init": {"seed": s, "theta": theta, "f": f},
                "alloc_settings": {
                    "method": "imse",
                    "use_Ki": True,
                    "theta": None,
                    "a0": None,
                    "gen": False,
                    "rho": rho,
                },
                "des_settings": {"is_explore": False, "is_exploit": True},
                "pc_settings": {"standardize": True, "latent": False},
                "trace": 0,
            },
        )

        Figure2(al_imse, theta_test, nugs, phat, "imse", ax[1])

        plt.savefig("Figure2.png", bbox_inches="tight")
        plt.show()
