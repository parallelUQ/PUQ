import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.stats import qmc
from PUQ.prior import prior_dist
from PUQ.designmethods.sequential_md_deterministic import sequential_design
from test_func import unimodal

if __name__ == "__main__":

    cex = unimodal()

    # # # Create a mesh for test set # # #
    xpl = np.linspace(cex.thetalimits[0][0], cex.thetalimits[0][1], 50)
    ypl = np.linspace(cex.thetalimits[1][0], cex.thetalimits[1][1], 50)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    thetatest = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
    ftest = np.zeros((thetatest.shape[0], 1))
    for i in range(thetatest.shape[0]):
        ftest[i, 0] = cex.function(thetatest[i, 0], thetatest[i, 1])

    ptest = sps.norm.pdf(cex.real_data - ftest, 0, np.sqrt(cex.obsvar))

    test_data = {"theta": thetatest, "f": ftest, "p": ptest, "p_prior": 1}
    # # # # # # # # # # # # # # # # # # # # #
    prior_func = prior_dist(dist="uniform")(
        a=cex.thetalimits[:, 0], b=cex.thetalimits[:, 1]
    )

    # Initial sample
    n0, s = 10, 2

    ndim = cex.thetalimits.shape[0]
    sampler = qmc.LatinHypercube(d=ndim, seed=s)
    # Generate samples in [0,1]^d
    unit_sample = sampler.random(n=n0)
    # Scale using limits
    t0 = qmc.scale(unit_sample, cex.thetalimits[:, 0], cex.thetalimits[:, 1])

    f0 = np.zeros((t0.shape[0], 1))
    for i in range(t0.shape[0]):
        f0[i, 0] = cex.function(t0[i, 0], t0[i, 1])

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i, af in enumerate(["rnd", "var", "ivar"]):
        des_obj = sequential_design(cex)
        des_obj.build_design(
            z0=t0,
            f0=f0,
            T=50,
            test=test_data,
            af=af,
            args={
                "mini_batch": 1,
                "n_init_thetas": 10,
                "nworkers": 2,
                "seed_n0": 1,
                "prior": prior_func,
                "data_test": None,
                "max_evals": 60,
                "type_init": None,
                "seed": 0,
                "integral": "LHS",
            },
        )

        theta_al = des_obj.zs

        cp = ax[i].contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap="RdGy")
        ax[i].scatter(
            theta_al[n0:, 0], theta_al[n0:, 1], c="black", marker="+", zorder=2
        )
        ax[i].scatter(
            theta_al[0:n0, 0],
            theta_al[0:n0, 1],
            zorder=2,
            marker="o",
            facecolors="none",
            edgecolors="blue",
        )
        ax[i].set_xlabel(r"$\theta_1$", fontsize=16)
        if i == 0:
            ax[i].set_ylabel(r"$\theta_2$", fontsize=16)

        ax[i].tick_params(axis="both", labelsize=16)
    plt.savefig("ex1.png", format="jpeg", bbox_inches="tight", dpi=1000)
    plt.show()
