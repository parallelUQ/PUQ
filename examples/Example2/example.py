import numpy as np
from ptest_funcs import sinfunc
import matplotlib.pyplot as plt
from scipy.stats import qmc
from PUQ.designmethods.sequential_1d_deterministic import sequential_design


if __name__ == "__main__":
    s = 1
    cex = sinfunc()
    dt = len(cex.true_theta)
    x_obs = np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None]
    cex.realdata(x=x_obs, seed=s)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    s = 1
    th_vec = [np.pi / 7, np.pi / 6, np.pi / 5, np.pi / 4]
    thlabel = [7, 6, 5, 4]
    x_vec = (np.arange(0, 100, 1) / 100)[:, None]
    fvec = np.zeros((len(th_vec), len(x_vec)))
    colors = ["blue", "orange", "red", "green", "purple"]
    for t_id, t in enumerate(th_vec):
        for x_id, x in enumerate(x_vec):
            fvec[t_id, x_id] = cex.function(x, t)[0]
        ax[0].plot(
            x_vec,
            fvec[t_id, :],
            label=r"$\theta=\pi/$" + str(thlabel[t_id]),
            color=colors[t_id],
            linewidth=3,
        )
    for d_id in range(len(cex.x)):
        ax[0].scatter(cex.x[d_id, 0], cex.real_data[0, d_id], color="black", s=50)
    ft = 16
    ax[0].set_xlabel(r"$x$", fontsize=ft)
    ax[0].set_ylabel(r"$\eta(x, \theta)$", fontsize=ft)
    ax[0].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
    ax[0].tick_params(labelsize=ft - 2)
    ax[0].legend(bbox_to_anchor=(1.1, -0.2), fontsize=ft - 2, ncol=4)

    # Generate initial sample
    n0, nmax = 10, 30
    ndim = cex.zlim.shape[0]
    sampler = qmc.LatinHypercube(d=ndim, seed=s)
    z0 = sampler.random(n=n0)
    f0 = np.array([cex.function(z0[i, 0], z0[i, 1]) for i in range(n0)])

    # Generate design
    des_obj = sequential_design(cex)
    des_obj.build_design(
        z0=z0,
        f0=f0[:, None],
        T=nmax,
        af="ivar",
        args={"nL": 200, "seed": s, "integral": "importance"},
    )

    ax[1].scatter(
        des_obj.zs[n0:, 0], des_obj.zs[n0:, 1], marker="+", c="red", s=200, linewidth=3
    )
    ax[1].hlines(
        cex.true_theta,
        0,
        1,
        linestyles="dotted",
        linewidth=3,
        colors="orange",
    )
    for xitem in x_obs:
        ax[1].vlines(
            xitem, 0, 1, linestyles="dotted", colors="orange", linewidth=3, zorder=1
        )
    ax[1].set_xlabel(r"$x$", fontsize=ft)
    ax[1].set_ylabel(r"$\theta$", fontsize=ft)
    ax[1].tick_params(labelsize=ft)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    plt.savefig("ex2.png", format="jpeg", bbox_inches="tight", dpi=1000)
    plt.show()
