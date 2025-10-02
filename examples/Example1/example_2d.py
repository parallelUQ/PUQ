import numpy as np
import scipy.stats as sps
from scipy.stats import qmc
from PUQ.designmethods.sequential_md_deterministic import sequential_design
from PUQ.prior import prior_dist
from test_func import unimodal, banana, bimodal, unidentifiable
import matplotlib.pyplot as plt

FUNCNAME = "banana"

if __name__ == "__main__":

    print("Running function: " + FUNCNAME)

    cex = eval(FUNCNAME)()

    # Create a mesh for test set
    xpl = np.linspace(cex.thetalimits[0][0], cex.thetalimits[0][1], 50)
    ypl = np.linspace(cex.thetalimits[1][0], cex.thetalimits[1][1], 50)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    thetatest = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
    ftest = np.zeros((thetatest.shape[0], cex.d))
    for i in range(thetatest.shape[0]):
        ftest[i, :] = cex.function(thetatest[i, 0], thetatest[i, 1])

    ptest = np.zeros(thetatest.shape[0])
    if cex.data_name == "unimodal":
        ptest = sps.norm.pdf(cex.real_data - ftest, 0, np.sqrt(cex.obsvar))
    else:
        for i in range(ftest.shape[0]):
            mean = ftest[i, :]
            rnd = sps.multivariate_normal(mean=mean, cov=cex.obsvar)
            ptest[i] = rnd.pdf(cex.real_data)
    test_data = {"theta": thetatest, "f": ftest, "p": ptest, "p_prior": 1}

    # Set a uniform prior
    prior_func = prior_dist(dist="uniform")(
        a=cex.thetalimits[:, 0], b=cex.thetalimits[:, 1]
    )

    # Define acquisition functions
    # acq_funcs = ["eivar", "rnd", "maxvar", "maxexp"]
    acq_funcs = ["ivar", "rnd", "var"]
    datalist = []
    rep_no, n0 = 2, 10

    # Run over 50 replications
    for seed_id in range(1, rep_no + 1):
        # Initial sample
        ndim = cex.thetalimits.shape[0]
        sampler = qmc.LatinHypercube(d=ndim, seed=seed_id)
        # Generate samples in [0,1]^d
        unit_sample = sampler.random(n=n0)
        # Scale using limits
        t0 = qmc.scale(unit_sample, cex.thetalimits[:, 0], cex.thetalimits[:, 1])
        f0 = np.zeros((t0.shape[0], cex.d))
        for i in range(t0.shape[0]):
            f0[i, :] = cex.function(t0[i, 0], t0[i, 1])

        for func in acq_funcs:
            print("Running " + func + " with seed " + str(seed_id))
            des_obj = sequential_design(cex)
            des_obj.build_design(
                z0=t0,
                f0=f0,
                T=75,
                test=test_data,
                af=func,
                args={
                    "mini_batch": 1,
                    "n_init_thetas": 10,
                    "nworkers": 2,
                    "prior": prior_func,
                    "data_test": test_data,
                    "seed": seed_id,
                    "integral": "LHS",
                },
            )

            theta_al = des_obj.zs

            fig, ax = plt.subplots()
            cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap="RdGy")
            ax.scatter(
                theta_al[n0:, 0], theta_al[n0:, 1], c="black", marker="+", zorder=2
            )
            ax.scatter(
                theta_al[0:n0, 0],
                theta_al[0:n0, 1],
                zorder=2,
                marker="o",
                facecolors="none",
                edgecolors="blue",
            )
            ax.set_xlabel(r"$\theta_1$", fontsize=16)
            ax.set_ylabel(r"$\theta_2$", fontsize=16)
            ax.tick_params(axis="both", labelsize=16)
            plt.show()
