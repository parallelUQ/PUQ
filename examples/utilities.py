import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from smt.sampling_methods import LHS


def twoD(designobj, Xpl, Ypl, p_test, nmesh):
    theta0 = designobj._info["theta0"]
    reps0 = designobj._info["reps0"]

    fig, ax = plt.subplots(figsize=(5, 5))
    cp = ax.contour(Xpl, Ypl, p_test.reshape(nmesh, nmesh), 20, cmap="RdGy")
    for label, x_count, y_count in zip(reps0, theta0[:, 0], theta0[:, 1]):
        if label == 2:
            plt.annotate(
                label,
                xy=(x_count, y_count),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=12,
                color="cyan",
                weight="bold",
            )
        else:
            plt.annotate(
                label,
                xy=(x_count, y_count),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=12,
                color="blue",
                weight="bold",
            )
    ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    plt.show()


def test_data_gen(cls_func, nmesh):
    xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
    ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
    if cls_func.data_name in [
        "unimodal",
        "branin",
        "himmelblau",
        "holder",
        "easom",
        "ackley",
        "sphere",
    ]:
        f_test = np.zeros(theta_test.shape[0])
    else:
        f_test = np.zeros((theta_test.shape[0], 2))
    p_test = np.zeros(theta_test.shape[0])
    for tid in range(0, len(theta_test)):
        if cls_func.data_name in [
            "unimodal",
            "branin",
            "himmelblau",
            "holder",
            "easom",
            "ackley",
            "sphere",
        ]:
            f_test[tid] = cls_func.function(theta_test[tid, 0], theta_test[tid, 1])
            rnd = sps.norm(loc=f_test[tid], scale=np.sqrt(cls_func.obsvar))
        else:
            f_test[tid, :] = cls_func.function(theta_test[tid, 0], theta_test[tid, 1])
            rnd = sps.multivariate_normal(mean=f_test[tid, :], cov=(cls_func.obsvar))
        p_test[tid] = rnd.pdf(cls_func.real_data)

    return theta_test, p_test, f_test, Xpl, Ypl


def test_data_gen_1d(cls_func, nmesh):

    # Create test data
    theta_test = np.linspace(
        cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh
    )[:, None]
    f_test, p_test = np.zeros(theta_test.shape[0]), np.zeros(theta_test.shape[0])
    for tid in range(0, len(theta_test)):
        f_test[tid] = cls_func.function(theta_test[tid, 0])
        rnd = sps.norm(loc=f_test[tid], scale=np.sqrt(cls_func.obsvar))
        p_test[tid] = rnd.pdf(cls_func.real_data)

    return theta_test, p_test, f_test


def oned(seqobject, x, obsvar, real_data, theta_test, f_test, p_test, cls_func, ninit):
    from PUQ.surrogate import emulator
    from scipy.stats import norm
    import matplotlib.pyplot as plt

    pc_settings = {"standardize": True, "latent": True}

    theta = seqobject._info["theta"][0:ninit, :]
    f = seqobject._info["f"][:, 0:ninit]

    theta_all = seqobject._info["theta"]
    f_all = seqobject._info["f"]

    theta0 = seqobject._info["theta0"]
    reps0 = seqobject._info["reps0"]

    emu = emulator(
        x=x,
        theta=theta,
        f=f,
        method="pcHetGP",
        args={
            "lower": None,
            "upper": None,
            "noiseControl": {
                "k_theta_g_bounds": (1, 100),
                "g_max": 1e2,
                "g_bounds": (1e-6, 1),
            },
            "init": {},
            "known": {},
            "settings": {
                "linkThetas": "joint",
                "logN": True,
                "initStrategy": "residuals",
                "checkHom": True,
                "penalty": True,
                "trace": 0,
                "return.matrices": True,
                "return.hom": False,
                "factr": 1e9,
            },
            "pc_settings": pc_settings,
        },
    )

    emupred = emu.predict(x=x, theta=theta_test)

    mean = emupred.mean()
    var = emupred.var()
    var_noisy = emupred._info["var_noisy"]

    # Probability corresponding to the quantile (e.g., 0.025 for the lower bound)
    quantile = 0.025
    lower_bound = norm.ppf(quantile, loc=mean, scale=np.sqrt(var))
    quantile = 0.975
    upper_bound = norm.ppf(quantile, loc=mean, scale=np.sqrt(var))
    # Probability corresponding to the quantile (e.g., 0.025 for the lower bound)
    quantile = 0.025
    lower_bound_nug = norm.ppf(quantile, loc=mean, scale=np.sqrt(var_noisy))
    quantile = 0.975
    upper_bound_nug = norm.ppf(quantile, loc=mean, scale=np.sqrt(var_noisy))

    ft = 20
    fig, ax = plt.subplots()
    ax.plot(theta_test, f_test, color="red")
    ax.plot(
        theta_test.flatten(),
        mean.flatten(),
        linestyle="dashed",
        color="blue",
        linewidth=2.5,
    )
    plt.fill_between(
        theta_test.flatten(),
        mean.flatten() - np.sqrt(var.flatten()),
        mean.flatten() + np.sqrt(var.flatten()),
        # lower_bound.flatten(),
        # upper_bound.flatten(),
        color="blue",
        alpha=0.3,
        linestyle="dotted",
    )

    # plt.fill_between(
    #     theta_test.flatten(),
    #     lower_bound_nug.flatten(),
    #     upper_bound_nug.flatten(),
    #     color='blue',
    #     alpha=0.1,
    #     linestyle="dotted",
    # )
    ax.scatter(
        theta.flatten(), f.flatten(), s=60, facecolors="none", edgecolors="green"
    )
    ax.set_xlabel(r"$\theta$", fontsize=ft)
    ax.set_ylabel(r"$M(\theta)$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.show()

    phat = np.zeros(theta_test.shape[0])
    phatvar = np.zeros(theta_test.shape[0])
    pvar1 = np.zeros(theta_test.shape[0])
    for tid in range(0, len(theta_test)):
        rnd = sps.norm(loc=mean[0, tid], scale=np.sqrt(obsvar + var[0, tid]))
        phat[tid] = rnd.pdf(real_data)
        rnd = sps.norm(loc=mean[0, tid], scale=np.sqrt(0.5 * obsvar + var[0, tid]))
        pvar1[tid] = rnd.pdf(real_data)
        phatvar[tid] = (1 / (2 * np.sqrt(np.pi) * np.sqrt(obsvar))) * pvar1[tid] - phat[
            tid
        ] ** 2

    fig, ax = plt.subplots()
    ax.plot(theta_test, p_test, color="red")
    ax.plot(theta_test, phat, color="blue", linestyle="dashed", linewidth=2.5)
    plt.fill_between(
        theta_test.flatten(),
        (phat - np.sqrt(phatvar)).flatten(),
        (phat + np.sqrt(phatvar)).flatten(),
        color="blue",
        alpha=0.1,
    )
    ax.set_xlabel(r"$\theta$", fontsize=ft)
    ax.set_ylabel(r"$p(y|\theta)$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.show()

    p_alloc = np.zeros(theta0.shape[0])
    f_alloc = np.zeros((1, theta0.shape[0]))
    for tid in range(0, len(theta0)):
        f_alloc[0, tid] = cls_func.function(theta0[tid])
        rnd = sps.norm(loc=f_alloc[0, tid], scale=np.sqrt(cls_func.obsvar))
        p_alloc[tid] = rnd.pdf(cls_func.real_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(theta_test, p_test, color="red", linewidth=2.5)
    for label, x_count, pval in zip(reps0, theta0, p_alloc):
        plt.annotate(
            label,
            xy=(x_count, pval),
            xytext=(0, 0),
            textcoords="offset points",
            color="blue",
            size=20,
        )
    ax.set_xlabel(r"$\theta$", fontsize=ft)
    ax.set_ylabel(r"$p(y|\theta)$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(theta0.flatten(), reps0.flatten(), color="blue", marker="*", s=55)
    ax.set_xlabel(r"$\theta$", fontsize=ft)
    ax.set_ylabel(r"a", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(
        theta_test.flatten(),
        emupred._info["nugs"].flatten(),
        color="blue",
        linestyle="dashed",
        linewidth=2.5,
    )
    ax.plot(theta_test.flatten(), cls_func.noise(theta_test).flatten(), color="black")
    ax.set_xlabel(r"$\theta$", fontsize=ft)
    ax.set_ylabel(r"$\mathbb{V}[\nu]$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.show()

    ft = 15

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, host = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    p1 = host.scatter(
        theta0.flatten(),
        reps0.flatten(),
        color="black",
        marker="*",
        s=55,
        label=r"$a_i$",
    )
    (p2,) = par1.plot(
        theta_test.flatten(),
        emupred._info["nugs"].flatten(),
        "b",
        linestyle="dashed",
        label="Variance",
        linewidth=2.5,
    )
    (p3,) = par2.plot(
        theta_test, phat, "r", linestyle="dotted", label="Likelihood", linewidth=2.5
    )

    host.set_xlim(-0.1, 1.1)
    host.set_ylim(0.9 * np.min(reps0.flatten()), 1.1 * np.max(reps0.flatten()))
    par1.set_ylim(
        0.9 * np.min(emupred._info["nugs"].flatten()),
        1.1 * np.max(emupred._info["nugs"].flatten()),
    )
    par2.set_ylim(-0.01, 1.1 * np.max(phat))

    host.set_xlabel(r"$\theta$", fontsize=ft)
    host.set_ylabel(r"$a_i$", fontsize=ft)
    par1.set_ylabel("Variance", fontsize=ft)
    par2.set_ylabel("Likelihood", fontsize=ft)
    host.tick_params(axis="both", labelsize=ft)
    par1.tick_params(axis="both", labelsize=ft)
    par2.tick_params(axis="both", labelsize=ft)
    lines = [p1, p2, p3]

    host.legend(
        lines,
        [l.get_label() for l in lines],
        loc="center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=3,
        prop={"size": ft},
    )

    plt.show()


def twodpaper(cls_func, Xpl, Ypl, p_test, theta0, reps0, thetainit=None, name=None):
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import ListedColormap

    yellow_colors = [
        (1, 1, 1),
        (1, 1, 0.8),  # light yellow
        (1, 1, 0.6),
        (1, 1, 0.4),
        (1, 1, 0.2),
        (1, 1, 0),  # yellow
        (1, 0.9, 0),  # dark yellow
        (1, 0.8, 0),  # yellow-orange
        (1, 0.6, 0),  # orange
        (1, 0.4, 0),  # dark orange
        (1, 0.2, 0),  # very dark orange
    ]
    yellow_cmap = ListedColormap(yellow_colors, name="yellow")

    if cls_func.data_name in ["unimodal", "branin"]:
        nmesh = len(Xpl)
        P = np.zeros((nmesh, nmesh))
        for i in range(nmesh):
            for j in range(nmesh):
                P[i, j] = cls_func.noise(
                    np.array([Xpl[i, j], Ypl[i, j]])[None, :]
                ).flatten()

        Pvar = P
    else:
        nmesh = len(Xpl)
        P = np.zeros((nmesh, nmesh, 2))
        for i in range(nmesh):
            for j in range(nmesh):
                P[i, j, :] = cls_func.noise(
                    np.array([Xpl[i, j], Ypl[i, j]])[None, :]
                ).flatten()
        Pvar = np.sum(P, axis=2)

    fig, ax = plt.subplots()
    cs = ax.contourf(Xpl, Ypl, Pvar, cmap=yellow_cmap, alpha=0.75)
    # cbar = fig.colorbar(cs, pad=0.1)
    cp = ax.contour(Xpl, Ypl, p_test.reshape(nmesh, nmesh), 20, cmap="coolwarm")

    if thetainit is None:
        for label, x_count, y_count in zip(reps0, theta0[:, 0], theta0[:, 1]):
            if label <= 2:
                plt.annotate(
                    label,
                    xy=(x_count, y_count),
                    xytext=(0, 0),
                    textcoords="offset points",
                    fontsize=12,
                    color="cyan",
                )
            else:
                plt.annotate(
                    label,
                    xy=(x_count, y_count),
                    xytext=(0, 0),
                    textcoords="offset points",
                    fontsize=12,
                    color="black",
                )
    else:
        for label, x_count, y_count in zip(reps0, theta0[:, 0], theta0[:, 1]):
            if np.array([x_count, y_count]) in thetainit:
                plt.annotate(
                    label,
                    xy=(x_count, y_count),
                    xytext=(0, 0),
                    textcoords="offset points",
                    fontsize=12,
                    color="cyan",
                )
            else:
                plt.annotate(
                    label,
                    xy=(x_count, y_count),
                    xytext=(0, 0),
                    textcoords="offset points",
                    fontsize=12,
                    color="black",
                )

    ax.set_xticks([0, 0.5, 1])  # Custom tick locations for x-axis
    ax.set_yticks([0, 0.5, 1])  # Custom tick locations for y-axis
    ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    plt.savefig(name, bbox_inches="tight")
    plt.show()


def twodpaperrev(cls_func, Xpl, Ypl, p_test, dictl, thetainit=None, name=None):
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import ListedColormap

    yellow_colors = [
        (1, 1, 1),
        (1, 1, 0.8),  # light yellow
        (1, 1, 0.6),
        (1, 1, 0.4),
        (1, 1, 0.2),
        (1, 1, 0),  # yellow
        (1, 0.9, 0),  # dark yellow
        (1, 0.8, 0),  # yellow-orange
        (1, 0.6, 0),  # orange
        (1, 0.4, 0),  # dark orange
        (1, 0.2, 0),  # very dark orange
    ]
    yellow_cmap = ListedColormap(yellow_colors, name="yellow")

    if cls_func.data_name in ["unimodal", "branin"]:
        nmesh = len(Xpl)
        P = np.zeros((nmesh, nmesh))
        for i in range(nmesh):
            for j in range(nmesh):
                P[i, j] = cls_func.noise(
                    np.array([Xpl[i, j], Ypl[i, j]])[None, :]
                ).flatten()

        Pvar = P
    else:
        nmesh = len(Xpl)
        P = np.zeros((nmesh, nmesh, 2))
        for i in range(nmesh):
            for j in range(nmesh):
                P[i, j, :] = cls_func.noise(
                    np.array([Xpl[i, j], Ypl[i, j]])[None, :]
                ).flatten()
        Pvar = np.sum(P, axis=2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for i in range(0, 3):
        if i == 0:
            for el in dictl:
                if el["method"] == "ivar":
                    reps0 = el["reps0"]
                    theta0 = el["theta0"]
        elif i == 1:
            for el in dictl:
                if el["method"] == "var":
                    reps0 = el["reps0"]
                    theta0 = el["theta0"]
        elif i == 2:
            for el in dictl:
                if el["method"] == "imse":
                    reps0 = el["reps0"]
                    theta0 = el["theta0"]

        cs = ax[i].contourf(Xpl, Ypl, Pvar, cmap=yellow_cmap, alpha=0.75)
        if i == 2:
            cbar = fig.colorbar(cs, ax=ax[i], pad=0.1)
        cp = ax[i].contour(Xpl, Ypl, p_test.reshape(nmesh, nmesh), 20, cmap="coolwarm")
        ft = 14
        if thetainit is None:
            for label, x_count, y_count in zip(reps0, theta0[:, 0], theta0[:, 1]):
                if label <= 2:
                    ax[i].annotate(
                        label,
                        xy=(x_count, y_count),
                        xytext=(0, 0),
                        textcoords="offset points",
                        fontsize=ft,
                        color="cyan",
                    )
                else:
                    ax[i].annotate(
                        label,
                        xy=(x_count, y_count),
                        xytext=(0, 0),
                        textcoords="offset points",
                        fontsize=ft,
                        color="black",
                    )
        else:
            for label, x_count, y_count in zip(reps0, theta0[:, 0], theta0[:, 1]):
                if np.array([x_count, y_count]) in thetainit:
                    ax[i].annotate(
                        label,
                        xy=(x_count, y_count),
                        xytext=(0, 0),
                        textcoords="offset points",
                        fontsize=ft,
                        color="cyan",
                    )
                else:
                    ax[i].annotate(
                        label,
                        xy=(x_count, y_count),
                        xytext=(0, 0),
                        textcoords="offset points",
                        fontsize=ft,
                        color="black",
                    )

        if i == 0:
            ax[i].set_yticks([0, 0.5, 1])  # Custom tick locations for x-axis
            ax[i].set_ylabel(r"$\theta_2$", fontsize=16)
        else:
            ax[i].set_yticks([])
        ax[i].set_xticks([0, 0.5, 1])

        ax[i].set_xlabel(r"$\theta_1$", fontsize=16)

        ax[i].tick_params(axis="both", labelsize=16)

    plt.savefig(name, bbox_inches="tight")
    plt.show()


def heatmap(cls_func):

    from matplotlib.colors import ListedColormap

    yellow_colors = [
        (1, 1, 1),
        (1, 1, 0.8),  # light yellow
        (1, 1, 0.6),
        (1, 1, 0.4),
        (1, 1, 0.2),
        (1, 1, 0),  # yellow
        (1, 0.9, 0),  # dark yellow
        (1, 0.8, 0),  # yellow-orange
        (1, 0.6, 0),  # orange
        (1, 0.4, 0),  # dark orange
        (1, 0.2, 0),  # very dark orange
    ]
    yellow_cmap = ListedColormap(yellow_colors, name="yellow")

    if cls_func.data_name in ["unimodal", "branin"]:
        nmesh = 50
        a = np.arange(nmesh + 1) / nmesh
        b = np.arange(nmesh + 1) / nmesh
        X, Y = np.meshgrid(a, b)
        Z = np.zeros((nmesh + 1, nmesh + 1))
        P = np.zeros((nmesh + 1, nmesh + 1))
        for i in range(nmesh + 1):
            for j in range(nmesh + 1):
                Z[i, j] = cls_func.function(X[i, j], Y[i, j])
                P[i, j] = cls_func.noise(
                    np.array([X[i, j], Y[i, j]])[None, :]
                ).flatten()

        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, P, cmap=yellow_cmap, alpha=0.75)
        cbar = fig.colorbar(cs)
        CS = ax.contour(X, Y, Z, colors="black")
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_xlabel(r"$\theta_1$", fontsize=16)
        ax.set_ylabel(r"$\theta_2$", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        plt.show()
    else:
        nmesh = 50
        a = np.arange(nmesh + 1) / nmesh
        b = np.arange(nmesh + 1) / nmesh
        X, Y = np.meshgrid(a, b)
        Z = np.zeros((nmesh + 1, nmesh + 1, 2))
        P = np.zeros((nmesh + 1, nmesh + 1, 2))

        for i in range(nmesh + 1):
            for j in range(nmesh + 1):
                Z[i, j, :] = cls_func.function(X[i, j], Y[i, j])
                P[i, j, :] = cls_func.noise(
                    np.array([X[i, j], Y[i, j]])[None, :]
                ).flatten()

        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, P[:, :, 0], cmap=yellow_cmap, alpha=0.75)
        cbar = fig.colorbar(cs)
        CS = ax.contour(X, Y, Z[:, :, 0], colors="black")
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_xlabel(r"$\theta_1$", fontsize=16)
        ax.set_ylabel(r"$\theta_2$", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        plt.show()

        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, P[:, :, 1], cmap=yellow_cmap, alpha=0.75)
        cbar = fig.colorbar(cs)
        CS = ax.contour(X, Y, Z[:, :, 1], colors="black")
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_xlabel(r"$\theta_1$", fontsize=16)
        ax.set_ylabel(r"$\theta_2$", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        plt.show()


def Figure1(f, theta, x, obsvar, real_data, theta_test, f_test, p_test, cls_func):
    from PUQ.surrogate import emulator
    from scipy.stats import norm
    import matplotlib.pyplot as plt

    pc_settings = {"standardize": True, "latent": False}

    # theta = seqobject._info['theta'][0:ninit,:]
    # f = seqobject._info['f'][:, 0:ninit]

    emu = emulator(
        x=x,
        theta=theta,
        f=f,
        method="pcHetGP",
        args={
            "lower": None,
            "upper": None,
            "noiseControl": {
                "k_theta_g_bounds": (1, 100),
                "g_max": 1e2,
                "g_bounds": (1e-6, 1),
            },
            "init": {},
            "known": {},
            "settings": {
                "linkThetas": "joint",
                "logN": True,
                "initStrategy": "residuals",
                "checkHom": True,
                "penalty": True,
                "trace": 0,
                "return.matrices": True,
                "return.hom": False,
                "factr": 1e9,
            },
            "pc_settings": pc_settings,
        },
    )

    emupred = emu.predict(x=x, theta=theta_test)

    mean = emupred.mean()
    var = emupred.var()
    var_noisy = emupred._info["var_noisy"]

    # Probability corresponding to the quantile (e.g., 0.025 for the lower bound)
    quantile = 0.025
    lower_bound = norm.ppf(quantile, loc=mean, scale=np.sqrt(var))
    quantile = 0.975
    upper_bound = norm.ppf(quantile, loc=mean, scale=np.sqrt(var))
    # Probability corresponding to the quantile (e.g., 0.025 for the lower bound)
    quantile = 0.025
    lower_bound_nug = norm.ppf(quantile, loc=mean, scale=np.sqrt(var_noisy))
    quantile = 0.975
    upper_bound_nug = norm.ppf(quantile, loc=mean, scale=np.sqrt(var_noisy))

    ft = 20
    fig, ax = plt.subplots()
    ax.plot(theta_test, f_test, color="red")
    ax.plot(
        theta_test.flatten(),
        mean.flatten(),
        linestyle="dashed",
        color="blue",
        linewidth=2.5,
    )
    plt.fill_between(
        theta_test.flatten(),
        mean.flatten() - np.sqrt(var.flatten()),
        mean.flatten() + np.sqrt(var.flatten()),
        # lower_bound.flatten(),
        # upper_bound.flatten(),
        color="blue",
        alpha=0.3,
        linestyle="dotted",
    )

    ax.scatter(
        theta.flatten(), f.flatten(), s=60, facecolors="none", edgecolors="green"
    )
    ax.set_xlabel(r"$\theta$", fontsize=ft)
    ax.set_ylabel(r"$\zeta(\theta)$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.savefig("Figure1a.png", bbox_inches="tight")
    plt.show()

    phat = np.zeros(theta_test.shape[0])
    phatvar = np.zeros(theta_test.shape[0])
    pvar1 = np.zeros(theta_test.shape[0])
    for tid in range(0, len(theta_test)):
        rnd = sps.norm(loc=mean[0, tid], scale=np.sqrt(obsvar + var[0, tid]))
        phat[tid] = rnd.pdf(real_data)
        rnd = sps.norm(loc=mean[0, tid], scale=np.sqrt(0.5 * obsvar + var[0, tid]))
        pvar1[tid] = rnd.pdf(real_data)
        phatvar[tid] = (1 / (2 * np.sqrt(np.pi) * np.sqrt(obsvar))) * pvar1[tid] - phat[
            tid
        ] ** 2

    fig, ax = plt.subplots()
    ax.plot(theta_test, p_test, color="red")
    ax.plot(theta_test, phat, color="blue", linestyle="dashed", linewidth=2.5)
    plt.fill_between(
        theta_test.flatten(),
        (phat - np.sqrt(phatvar)).flatten(),
        (phat + np.sqrt(phatvar)).flatten(),
        color="blue",
        alpha=0.1,
    )
    ax.set_xlabel(r"$\theta$", fontsize=ft)
    ax.set_ylabel(r"$p(y|\theta)$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.savefig("Figure1b.png", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(
        theta_test.flatten(),
        emupred._info["nugs"].flatten(),
        color="blue",
        linestyle="dashed",
        linewidth=2.5,
    )
    ax.plot(theta_test.flatten(), cls_func.noise(theta_test).flatten(), color="black")
    ax.set_xlabel(r"$\theta$", fontsize=ft)
    ax.set_ylabel(r"$\mathbb{V}[\nu]$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.savefig("Figure1c.png", bbox_inches="tight")
    plt.show()

    return theta_test, emupred._info["nugs"].flatten(), phat


def Figure2(desobject, theta_test, nugs, phat, method):

    theta0 = desobject._info["theta0"]
    reps0 = desobject._info["reps0"]

    ft = 15

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, host = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    par2.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)

    p1 = host.scatter(
        theta0.flatten(),
        reps0.flatten(),
        color="black",
        marker="*",
        s=55,
        label=r"$a_i$",
    )
    (p2,) = par1.plot(
        theta_test.flatten(),
        nugs,
        "b",
        linestyle="dashed",
        label="Variance",
        linewidth=2.5,
    )
    (p3,) = par2.plot(
        theta_test, phat, "r", linestyle="dotted", label="Likelihood", linewidth=2.5
    )

    host.set_xlim(-0.1, 1.1)
    host.set_ylim(0.8 * np.min(reps0.flatten()), 1.1 * np.max(reps0.flatten()))
    par1.set_ylim(0.8 * np.min(nugs), 1.1 * np.max(nugs))
    par2.set_ylim(-0.1, 1.1 * np.max(phat))

    host.set_xlabel(r"$\theta$", fontsize=ft)
    host.set_ylabel(r"$a_i$", fontsize=ft)
    par1.set_ylabel("Variance", fontsize=ft)
    par2.set_ylabel("Likelihood", fontsize=ft)
    host.tick_params(axis="both", labelsize=ft)
    par1.tick_params(axis="both", labelsize=ft)
    par2.tick_params(axis="both", labelsize=ft)
    lines = [p1, p2, p3]

    host.legend(
        lines,
        [l.get_label() for l in lines],
        loc="center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=3,
        prop={"size": ft},
    )
    plt.savefig("Figure2" + method + ".png", bbox_inches="tight")
    plt.show()
