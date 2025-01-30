from PUQ.utils import read_output
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen, twoD
import scipy.stats as sps


def read_data(
    rep0=0,
    repf=10,
    methods=["ivar", "imse", "unif"],
    batches=[8, 16, 32, 64],
    examples=["unimodal", "banana", "bimodal"],
    ids=["3", "1", "2"],
    ee="exploit",
    metric="TV",
    folderpath=None,
    ntotal=192,
    initial=30,
):

    datalist1, datalist2 = [], []
    for eid, example in enumerate(examples):
        for m in methods:
            for bid, b in enumerate(batches):
                path = (
                    folderpath
                    + ids[eid]
                    + "_"
                    + example
                    + "_"
                    + ee
                    + "/"
                    + str(b)
                    + "/"
                )
                for r in range(rep0[eid], repf[eid]):
                    desobj = read_output(path, example, m, b + 1, b, r)
                    reps0 = desobj._info["reps0"]
                    theta0 = desobj._info["theta0"]

                    f = desobj._info["f"]

                    if np.unique(f, axis=1).shape[1] != (ntotal[eid] + initial[eid]):
                        print(np.unique(f, axis=1).shape)

                    if metric == "TV":
                        for tvid, tv in enumerate(desobj._info["TV"]):
                            datalist1.append(
                                {
                                    "MAD": tv,
                                    "t": tvid,
                                    "rep": r,
                                    "batch": b,
                                    "worker": b + 1,
                                    "method": m,
                                    "example": example,
                                }
                            )
                    else:
                        for tvid, tv in enumerate(desobj._info["TViter"]):
                            if tvid < ntotal[eid] + 1:
                                datalist1.append(
                                    {
                                        "MAD": tv,
                                        "t": tvid,
                                        "rep": r,
                                        "batch": b,
                                        "worker": b + 1,
                                        "method": m,
                                        "example": example,
                                    }
                                )

                    datalist2.append(
                        {
                            "rep": r,
                            "batch": b,
                            "worker": b + 1,
                            "method": m,
                            "example": example,
                            "iter_explore": desobj._info["iter_explore"],
                            "iter_exploit": desobj._info["iter_exploit"],
                        }
                    )

    df1 = pd.DataFrame(datalist1)
    df2 = pd.DataFrame(datalist2)
    return df1, df2


def lineplot(df, examples, batches, metric="TV", ci=None, label=None):
    
    le = len(examples)
    fig, ax = plt.subplots(1, le, figsize=(8*le, 6))
    ft = 20
    for i, example in enumerate(examples):
        df1 = df.loc[df["example"] == example]
        print(df1.shape)
        sns.lineplot(
            data=df1,
            x="t",
            y="MAD",
            hue="method",
            style="batch",
            palette=["r", "g", "b", "c"],
            errorbar=ci,
            linewidth=5,
            ax=ax[i],
        )
        if example in ["bimodal", "SEIRDS"]:
            lgd = ax[i].legend(
                loc="upper center",
                bbox_to_anchor=(1.2, 0.8),
                fancybox=True,
                shadow=True,
                ncol=1,
                fontsize=ft - 5,
            )
        else:
            ax[i].legend([], [], frameon=False)
        ax[i].set_yscale("log")
        if metric == "TV":
            ax[i].set_xlabel("t", fontsize=ft)
        else:
            ax[i].set_xlabel("# of simulation evaluations", fontsize=ft)
        ax[i].set_ylabel("MAD", fontsize=ft)
        ax[i].tick_params(axis="both", labelsize=ft)
    if label is not None:
        plt.savefig(label, bbox_inches='tight')
    plt.show()



def exp_ratio(dfexpl, examples, methods, batches, ntotals, label):
    
    ft = 20
    vals = []
    for mid, m in enumerate(methods):
        print(m)
        for i, ex in enumerate(examples):
            ntotal = ntotals[i]
            for bid, b in enumerate(batches):
                dffilt = dfexpl.loc[
                    (dfexpl["method"] == m)
                    & (dfexpl["example"] == ex)
                    & (dfexpl["batch"] == b)
                ]
                vals.extend(
                    {"method": m, "b": b, "example": ex, "exploration": e}
                    for e in dffilt["iter_explore"] / (ntotal / b)
                )

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    df = pd.DataFrame(vals)
    df["percent"] = df["exploration"] * 100
    for i in range(0, 3):
        dfm = df[df["method"] == methods[i]]
        print(dfm.shape)
        sns.boxplot(
            x="example", y="percent", hue="b", data=dfm, ax=ax[i], showfliers=False
        )
        if i == 0:
            ax[i].set_ylabel("Exploration stages (%)", fontsize=ft - 3)
        else:
            ax[i].set_ylabel("")
        ax[i].set_xlabel("Example", fontsize=ft - 3)
        ax[i].tick_params(axis="both", labelsize=ft - 5)
        if i == 1:
            lgd = ax[i].legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.5),
                fancybox=True,
                shadow=True,
                ncol=4,
                fontsize=ft - 5,
                title="b",
                title_fontsize=ft - 5,
            )
        else:
            ax[i].legend_.remove()
        ax[i].set_ylim(0, 101)
        ax[i].axhline(y=50, color="red", ls="--", lw=5)
    plt.savefig(label, bbox_inches='tight')
    plt.show()



def interval_score(
    examples, methods, batches, rep0, repf, initial, ids, folderpath, ee
):

    for exid, ex in enumerate(examples):
        result = []
        for r in range(rep0[exid], repf[exid]):
            cls_func = eval(ex)()
            cls_func.realdata(seed=r)
            nmesh = 50
            theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
            thetamle = theta_test[np.argmax(p_test), :]
            # print(thetamle)
            for mid, m in enumerate(methods):
                for bid, b in enumerate(batches):
                    path = (
                        folderpath
                        + ids[exid]
                        + "_"
                        + ex
                        + "_"
                        + ee
                        + "/"
                        + str(b)
                        + "/"
                    )
                    desobj = read_output(path, ex, m, b + 1, b, r)
                    theta = desobj._info["theta"][initial[exid] :, :]
                    for i in range(theta.shape[1]):
                        total_is = compute_interval_score(theta[:, i], thetamle[i])
                        result.append(
                            {
                                "i": i,
                                "score": total_is,
                                "method": m,
                                "batch": b,
                                "example": ex,
                            }
                        )

        print(ex)
        df = pd.DataFrame(result)
        for mid, m in enumerate(methods):
            print(m)
            for i in range(theta.shape[1]):
                print("\u03b8", end=", ")
                for bid, b in enumerate(batches):
                    df1 = df.loc[df["method"] == m]
                    df2 = df1.loc[df["batch"] == b]
                    df3 = df2.loc[df["i"] == i]
                    print(np.round(np.median(df3["score"]), 1), end=" ")
                    print("(" + str(np.round(np.std(df3["score"]), 1)) + ")", end=", ")
                print()

def compute_interval_score(theta, thetamle):
    alpha = 0.1
    u = np.quantile(theta, 1 - alpha / 2)
    l = np.quantile(theta, alpha / 2)

    is_l = 1 if thetamle < l else 0
    is_u = 1 if thetamle > u else 0

    total_is = (
        (u - l)
        + (2 / alpha) * (l - thetamle) * (is_l)
        + (2 / alpha) * (thetamle - u) * (is_u)
    )

    return total_is


def interval_score_SIR(
    examples, methods, batches, rep0, repf, initial, ids, folderpath, ee
):
    from smt.sampling_methods import LHS
    from sir_funcs import SIR, SEIRDS

    for exid, ex in enumerate(examples):

        # # Create test data
        n0 = initial[exid]
        nmesh = 50
        nt = nmesh**2
        nrep = 1000
        cls_func = eval(ex)()
        if ex == "SIR":
            xpl = np.linspace(
                cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh
            )
            ypl = np.linspace(
                cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh
            )
            Xpl, Ypl = np.meshgrid(xpl, ypl)
            theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
        elif ex == "SEIRDS":
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=100)
            theta_test = sampling(nt)

        f_test = np.zeros((nt, cls_func.d))
        persis_info = {"rand_stream": np.random.default_rng(100)}
        for thid, th in enumerate(theta_test):
            IrIdRD = cls_func.sim_f(
                thetas=th, return_all=True, repl=nrep, persis_info=persis_info
            )
            f_test[thid, :] = np.mean(IrIdRD, axis=0)

        result = []
        for r in range(rep0[exid], repf[exid]):
            cls_func = eval(ex)()
            cls_func.realdata(seed=r)
            p_test = np.zeros(nmesh**2)
            for thid, th in enumerate(theta_test):
                rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
                p_test[thid] = rnd.pdf(cls_func.real_data)
            thetamle = theta_test[np.argmax(p_test), :]
            for mid, m in enumerate(methods):
                for bid, b in enumerate(batches):
                    path = (
                        folderpath
                        + ids[exid]
                        + "_"
                        + ex
                        + "_"
                        + ee
                        + "/"
                        + str(b)
                        + "/"
                    )
                    desobj = read_output(path, ex, m, b + 1, b, r)
                    theta = desobj._info["theta"][n0:, :]
                    for i in range(theta.shape[1]):
                        total_is = compute_interval_score(theta[:, i], thetamle[i])
                        result.append(
                            {
                                "i": i,
                                "score": total_is,
                                "method": m,
                                "batch": b,
                                "example": ex,
                            }
                        )

        df = pd.DataFrame(result)
        for mid, m in enumerate(methods):
            print(m)
            for i in range(theta.shape[1]):
                print("\u03b8", end=", ")
                for bid, b in enumerate(batches):
                    df1 = df.loc[df["method"] == m]
                    df2 = df1.loc[df["batch"] == b]
                    df3 = df2.loc[df["i"] == i]
                    print(np.round(np.median(df3["score"]), 1), end=" ")
                    print("(" + str(np.round(np.std(df3["score"]), 1)) + ")", end=", ")
                print()


def SIR2D(example, batch, method, r, ids=None, ee=None, folderpath=None):
    from sir_funcs import SIR
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
    path = folderpath + ids + "_" + example + "_" + ee + "/" + str(batch) + "/"
    yellow_cmap = ListedColormap(yellow_colors, name="yellow")
    m_list = ["ivar", "var", "imse"]
    theta_list, rep_list = [], []
    for m in m_list:
        desobj = read_output(path, example, m, batch + 1, batch, r)
        theta_m = desobj._info["theta0"]
        reps_m = desobj._info["reps0"]
        theta_list.append(theta_m)
        rep_list.append(reps_m)
        print("Iter explore")
        print(desobj._info["iter_explore"])
        print("Iter exploit")
        print(desobj._info["iter_exploit"])

    # Create test data
    nrep = 1000
    persis_info = {"rand_stream": np.random.default_rng(100)}
    cls_func = eval("SIR")()
    cls_func.realdata(seed=r)

    from smt.sampling_methods import LHS

    n0 = 15
    sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(r))
    thetainit = sampling(n0)
    nmesh = 50
    nt = nmesh**2
    xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
    ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
    f_test = np.zeros((nt, cls_func.d))
    f_var = np.zeros((nt, cls_func.d))
    for thid, th in enumerate(theta_test):
        IrIdRD = cls_func.sim_f(
            thetas=th, return_all=True, repl=nrep, persis_info=persis_info
        )
        f_test[thid, :] = np.mean(IrIdRD, axis=0)
        f_var[thid, :] = np.var(IrIdRD, axis=0)

    p_test = np.zeros(nmesh**2)
    for thid, th in enumerate(theta_test):
        rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
        p_test[thid] = rnd.pdf(cls_func.real_data)

    ft = 18
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for i in range(0, 3):
        reps0 = rep_list[i]
        theta0 = theta_list[i]

        cs = ax[i].contourf(
            Xpl,
            Ypl,
            np.sum(f_var, axis=1).reshape(nmesh, nmesh),
            cmap=yellow_cmap,
            alpha=0.75,
        )
        if i == 2:
            cbar = fig.colorbar(cs, ax=ax[i], pad=0.1)
            
        cp = ax[i].contour(Xpl, Ypl, p_test.reshape(nmesh, nmesh), 20, cmap="coolwarm")
        for label, x_count, y_count in zip(reps0, theta0[:, 0], theta0[:, 1]):
            if np.array([x_count, y_count]) in thetainit:
                co = "cyan"
            else:
                co = "black"
            ax[i].annotate(
                label,
                xy=(x_count, y_count),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=ft - 2,
                color=co,
            )

        if i == 0:
            ax[i].set_yticks([0, 0.5, 1])  # Custom tick locations for y-axis
            ax[i].set_ylabel(r"$\theta_2$", fontsize=ft)
        else:
            ax[i].set_yticks([])

        ax[i].set_xticks([0, 0.5, 1])  # Custom tick locations for x-axis
        ax[i].set_xlabel(r"$\theta_1$", fontsize=ft)
        ax[i].tick_params(axis="both", labelsize=ft)
    plt.savefig('Figure12_rev.png', bbox_inches='tight')
    plt.show()


def SIRfuncevals(example, batch, r, ids, ee, initial, folderpath):
    from sir_funcs import SIR
    
    m_list = ["ivar", "var", "imse"]
    theta_list = []
    labs = ["Susceptible", "Infected", "Recovered"]
    ft = 18
    lw = 3
    path = folderpath + ids + "_" + example + "_" + ee + "/" + str(batch) + "/"

    for m in m_list:
        desobj = read_output(path, example, m, batch + 1, batch, r)
        thetas_m = desobj._info["theta"][initial:,]
        theta_list.append(thetas_m)

    cls_func = eval(example)()
    persis_info = {"rand_stream": np.random.default_rng(100)}
    Strue, Itrue, Rtrue = cls_func.simulation(
        thetas=cls_func.theta_true, repl=1000, persis_info=persis_info
    )
            
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(0, 3):
        thetas = theta_list[i]

        for th in thetas:
            S, I, R = cls_func.simulation(thetas=th, repl=1000, persis_info=persis_info)

            axs[0, i].plot(np.mean(S, axis=1), c="orange", alpha=0.5)
            axs[1, i].plot(np.mean(I, axis=1), c="pink", alpha=0.5)
            axs[2, i].plot(np.mean(R, axis=1), c="cyan", alpha=0.5)

        for j in range(0, 2):
            axs[j, i].tick_params(
                axis="x", which="both", length=0
            )  # Hide tick marks
            axs[j, i].set_xticks([])  # Remove x-axis tick labels

        if i > 0:
            for j in range(0, 3):
                axs[j, i].tick_params(
                    axis="y", which="both", length=0
                )  # Hide tick marks
                axs[j, i].set_yticks([])  # Remove x-axis tick labels

        axs[0, i].plot(
            np.mean(Strue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )
        axs[1, i].plot(
            np.mean(Itrue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )
        axs[2, i].plot(
            np.mean(Rtrue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )
        axs[0, i].set_ylim(0, 1000)
        axs[1, i].set_ylim(0, 500)
        axs[2, i].set_ylim(0, 1000)
        axs[0, i].set_xlim(0, 150)
        axs[1, i].set_xlim(0, 150)
        axs[2, i].set_xlim(0, 150)
        axs[i, 0].set_ylabel(labs[i], fontsize=ft)
        axs[i, 0].tick_params(axis="y", labelsize=ft - 2)
        axs[2, i].set_xlabel("Time", fontsize=ft)
        axs[2, i].tick_params(axis="x", labelsize=ft - 2)
    plt.savefig('Figure13_rev.png', bbox_inches='tight')
    plt.show()


def SEIRDSfuncevals(example, batch, r, ids, ee, initial, folderpath):
    from sir_funcs import SEIRDS

    ft = 18
    lw = 3
    m_list = ["ivar", "var", "imse"]
    theta_list = []
    labs = ["Susceptible", "Exposed", "Infected (Recover)", "Infected (Dead)", "Recovered", "Dead"]
    path = folderpath + ids + "_" + example + "_" + ee + "/" + str(batch) + "/"
    
    for m in m_list:
        desobj = read_output(path, example, m, batch + 1, batch, r)
        thetas_m = desobj._info["theta"][initial:,]
        theta_list.append(thetas_m)

    cls_func = eval(example)()
    persis_info = {"rand_stream": np.random.default_rng(100)}
    Strue, Etrue, Irtrue, Idtrue, Rtrue, Dtrue = cls_func.simulation(
        thetas=cls_func.theta_true, repl=1000, persis_info=persis_info
    )
        
    fig, axs = plt.subplots(6, 3, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(0, 3):
        thetas = theta_list[i]

        for th in thetas:

            S, E, Ir, Id, R, D = cls_func.simulation(
                thetas=th, repl=1000, persis_info=persis_info
            )

            axs[0, i].plot(np.mean(S, axis=1), c="orange", alpha=0.3)
            axs[1, i].plot(np.mean(E, axis=1), c="pink", alpha=0.3)
            axs[2, i].plot(np.mean(Ir, axis=1), c="cyan", alpha=0.3)
            axs[3, i].plot(np.mean(Id, axis=1), c="violet", alpha=0.3)
            axs[4, i].plot(np.mean(R, axis=1), c="yellow", alpha=0.3)
            axs[5, i].plot(np.mean(D, axis=1), c="lime", alpha=0.3)

        axs[0, i].plot(
            np.mean(Strue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )
        axs[1, i].plot(
            np.mean(Etrue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )
        axs[2, i].plot(
            np.mean(Irtrue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )
        axs[3, i].plot(
            np.mean(Idtrue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )
        axs[4, i].plot(
            np.mean(Rtrue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )
        axs[5, i].plot(
            np.mean(Dtrue, axis=1), c="black", linestyle="dotted", linewidth=lw
        )

        axs[0, i].set_ylim(0, 1000)
        axs[1, i].set_ylim(0, 200)
        axs[2, i].set_ylim(0, 300)
        axs[3, i].set_ylim(0, 500)
        axs[4, i].set_ylim(0, 500)
        axs[5, i].set_ylim(0, 1000)

        axs[0, i].set_xlim(0, 150)
        axs[1, i].set_xlim(0, 150)
        axs[2, i].set_xlim(0, 150)
        axs[3, i].set_xlim(0, 150)
        axs[4, i].set_xlim(0, 150)
        axs[5, i].set_xlim(0, 150)

        for j in range(0, 5):
            axs[j, i].tick_params(axis="x", which="both", length=0)  # Hide tick marks
            axs[j, i].set_xticks([])  # Remove x-axis tick labels
        
        for j in range(0, 6):
            axs[j, 0].set_ylabel(labs[j], fontsize=ft)
            axs[j, 0].tick_params(axis="y", labelsize=ft - 2)

        if i > 0:
            for j in range(0, 6):
                axs[j, i].tick_params(
                    axis="y", which="both", length=0
                )  # Hide tick marks
                axs[j, i].set_yticks([])  # Remove x-axis tick labels
        
        axs[5, i].set_xlabel("Time", fontsize=ft)
        axs[5, i].tick_params(axis="x", labelsize=ft - 2)
    plt.savefig('FigureE3_rev.png', bbox_inches='tight')
    plt.show()


def boxplot(df, examples, batches):
    for example in examples:
        for b in batches:
            df1 = df.loc[df["example"] == example]
            df2 = df1.loc[df1["batch"] == b]
            sns.boxplot(
                x=df2["t"],
                y=df2["MAD"],
                hue=df2["method"],
                showfliers=False,
                palette="Set2",
            ).set_title(example + "_" + str(b))
            plt.show()


def boxplot_batch(df, examples, methods):
    for example in examples:
        for m in methods:
            df1 = df.loc[df["example"] == example]
            df2 = df1.loc[df["method"] == m]
            fig, ax = plt.subplots()
            # fig.set_size_inches(20, 5)
            values_list = [0, 64, 128, 192, 256]
            dfnew = df2[df2["t"].isin(values_list)]
            sns.boxplot(
                x=dfnew["t"],
                y=dfnew["MAD"],
                hue=dfnew["batch"],
                showfliers=False,
                palette="Set2",
            ).set_title(example + "_" + m)
            plt.show()
            
def SIRtheta(example, batch, method, r, initial, ids=None, ee=None, folderpath=None):
    cls_func = eval(str(example))()
    p = cls_func.p
    path = folderpath + ids + "_" + example + "_" + ee + "/" + str(batch) + "/"

    desobj = read_output(path, example, method, batch + 1, batch, r)
    print(desobj._info["iter_explore"])
    print(desobj._info["iter_exploit"])
    thetas = desobj._info["theta"]
    pdtheta = pd.DataFrame(thetas)

    pdtheta["color"] = np.concatenate(
        (np.repeat("red", initial), np.repeat("gray", 256))
    )
    labs = [
        r"$\theta_1$",
        r"$\theta_2$",
        r"$\theta_3$",
        r"$\theta_4$",
        r"$\theta_5$",
        r"$\theta_6$",
        r"$\theta_7$",
    ]
    sns.set_theme(style="white")
    g = sns.pairplot(
        pdtheta,
        kind="scatter",
        diag_kind="hist",
        corner=True,
        hue="color",
        palette=["blue", "gray"],
        markers=["*", "X"],
    )

    ft = 20
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import FormatStrFormatter

    for i in range(0, p):
        for j in range(0, i + 1):
            g.axes[i, j].set_xlim((0, 1))

    for j in range(0, p):
        g.axes[j, j].axvline(
            x=cls_func.theta_true[j], color="red", linestyle="--", lw=2
        )

    for j in range(0, p):
        lab = "\theta_" + str(j)
        g.axes[j, 0].set_ylabel(labs[j], fontsize=ft)
        g.axes[p - 1, j].set_xlabel(labs[j], fontsize=ft)

    for j in range(1, p):
        g.axes[j, 0].yaxis.set_major_locator(MaxNLocator(3))
        g.axes[j, 0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        g.axes[j, 0].tick_params(axis="both", which="major", labelsize=ft - 2)

    for j in range(0, p):
        g.axes[p - 1, j].xaxis.set_major_locator(MaxNLocator(3))
        g.axes[p - 1, j].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        g.axes[p - 1, j].tick_params(axis="both", which="major", labelsize=ft - 2)

    g._legend.remove()
    # plt.savefig("Figure13_" + method + ".png", bbox_inches="tight")
    plt.show

def visual_theta(example, method, batch, rep0, repf, initial, folderpath, ids, ee):

    path = folderpath + ids + "_" + example + "_" + ee + "/" + str(batch) + "/"
    print(path)
    for r in range(rep0, repf):
        desobj = read_output(path, example, method, batch + 1, batch, r)
        reps0 = desobj._info["reps0"]
        theta0 = desobj._info["theta0"]

        f = desobj._info["f"]
        theta = desobj._info["theta"]
        print(np.unique(f, axis=1).shape)
        print(np.unique(theta, axis=0).shape)

        nmesh = 50
        cls_func = eval(example)()
        cls_func.realdata(seed=None)

        theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
        twoD(desobj, Xpl, Ypl, p_test, nmesh)