from plotutils import plotresult, plotparams
import matplotlib.pyplot as plt
import numpy as np

clist = ["b", "r", "g", "m", "y", "c", "pink", "purple"]
mlist = ["P", "p", "*", "o", "s", "h"]
linelist = ["-", "--", "-.", ":", "-.", ":"]

labelsb = ["EI", "EIVAR", "HYBRID", "RND"]
method = ["ei", "eivar", "hybrid_ei", "rnd"]
example_name = ["sphere", "matyas", "ackley"]
example_name = ["himmelblau", "holder", "easom"]

batch = 1
worker = 2
rep = 30
fonts = 22

path = "/Users/ozgesurer/Desktop/sh_files/"

for metric in ["AE", "MAD"]:
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    for exid, ex in enumerate(example_name):
        for mid, m in enumerate(method):
            out = ex + "_" + m
            avgAE, avgtime, avgTV = plotresult(
                path, out, ex, worker, batch, rep, m, n0=10, nf=1000
            )
            if metric == "AE":
                axes[exid].plot(
                    np.arange(len(avgAE)),
                    avgAE,
                    label=labelsb[mid],
                    color=clist[mid],
                    linestyle=linelist[mid],
                )
            else:
                axes[exid].plot(
                    np.arange(len(avgTV)),
                    avgTV,
                    label=labelsb[mid],
                    color=clist[mid],
                    linestyle=linelist[mid],
                )
        axes[exid].set_yscale("log")
        # axes[exid].set_xscale('log')
        axes[exid].set_xlabel("# of parameters", fontsize=fonts)
        if exid == 0:
            if metric == "AE":
                axes[exid].set_ylabel(r"$\delta$", fontsize=fonts)
            else:
                axes[exid].set_ylabel(r"MAD", fontsize=fonts)
        axes[exid].tick_params(axis="both", which="major", labelsize=fonts - 5)

    axes[1].legend(bbox_to_anchor=(1.3, -0.2), ncol=4, fontsize=fonts)
    plt.show()

show1 = False
if show1:
    labelsb = ["10", "100", "1000", "RND"]
    method = ["hybrid_ei", "hybrid_ei", "hybrid_ei", "rnd"]
    outs = [10, 100, 1000]
    example_name = ["himmelblau"]

    for metric in ["AE"]:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        for exid, ex in enumerate(example_name):
            for mid, m in enumerate(method):
                if mid < 3:
                    out = ex + "_" + m + "_" + str(outs[mid])
                else:
                    out = ex + "_" + m
                avgAE, avgtime, avgTV = plotresult(
                    path, out, ex, worker, batch, rep, m, n0=10, nf=1000
                )
                if metric == "AE":
                    axes.plot(
                        np.arange(len(avgAE)),
                        avgAE,
                        label=labelsb[mid],
                        color=clist[mid],
                        linestyle=linelist[mid],
                    )
                else:
                    axes.plot(
                        np.arange(len(avgTV)),
                        avgTV,
                        label=labelsb[mid],
                        color=clist[mid],
                        linestyle=linelist[mid],
                    )
            axes.set_yscale("log")
            axes.set_xscale("log")
            axes.set_xlabel("# of parameters", fontsize=fonts)
            if exid == 0:
                if metric == "AE":
                    axes.set_ylabel(r"$\delta$", fontsize=fonts)
                else:
                    axes.set_ylabel(r"MAD", fontsize=fonts)
            axes.tick_params(axis="both", which="major", labelsize=fonts - 5)

        axes.legend(bbox_to_anchor=(1.3, -0.2), ncol=4, fontsize=fonts)
        plt.show()


show2 = True
if show2:

    clist = ["b", "r", "g", "m", "y", "c", "pink", "purple"]
    mlist = ["P", "p", "*", "o", "s", "h"]
    linelist = ["-", "--", "-.", ":", "-.", ":"]

    labelsb = ["b=1", "b=32", "b=64", "b=128"]
    method = ["ei"]
    batch_sizes = [1, 32, 64, 128]
    batch_sizes = [1, 5, 25, 125]
    example_name = ["sphere", "matyas", "ackley"]
    # example_name = ['matyas']
    # example_name = ['himmelblau', 'holder', 'easom']
    worker = 126
    rep = 28
    fonts = 22

    path = "/Users/ozgesurer/Desktop/sh_files/batch_mode/"

    for metric in ["AE", "MAD"]:
        fig, axes = plt.subplots(1, 3, figsize=(22, 5))
        for exid, ex in enumerate(example_name):
            for bid, b in enumerate(batch_sizes):
                for mid, m in enumerate(method):
                    out = ex + "_" + m + "_" + "b" + str(b) + "_" + "w128"
                    avgAE, avgtime, avgTV = plotresult(
                        path, out, ex, worker, b, rep, m, n0=128, nf=1000
                    )
                    if metric == "AE":
                        axes[exid].plot(
                            np.arange(len(avgAE)),
                            avgAE,
                            label=labelsb[bid],
                            color=clist[bid],
                            linestyle=linelist[bid],
                        )
                    else:
                        axes[exid].plot(
                            np.arange(len(avgTV)),
                            avgTV,
                            label=labelsb[bid],
                            color=clist[bid],
                            linestyle=linelist[bid],
                        )
            axes[exid].set_yscale("log")
            # axes[exid].set_xscale('log')
            axes[exid].set_xlabel("# of parameters", fontsize=fonts)
            if exid == 0:
                if metric == "AE":
                    axes[exid].set_ylabel(r"$\delta$", fontsize=fonts)
                else:
                    axes[exid].set_ylabel(r"MAD", fontsize=fonts)
            axes[exid].tick_params(axis="both", which="major", labelsize=fonts - 5)

        axes[1].legend(bbox_to_anchor=(1.3, -0.2), ncol=4, fontsize=fonts)
        plt.show()


show3 = False
if show3:

    clist = ["b", "r", "g", "m", "y", "c", "pink", "purple"]
    mlist = ["P", "p", "*", "o", "s", "h"]
    linelist = ["-", "--", "-.", ":", "-.", ":"]

    labelsb = ["b=1", "b=32", "b=64", "b=128"]
    m = "ei"

    ex = "easom"
    worker = 129
    rep = 30
    fonts = 22

    path = "/Users/ozgesurer/Desktop/sh_files/batch_mode/"
    b = 32
    out = ex + "_" + m + "_" + "b" + str(b) + "_" + "w128"
    plotparams(path, out, ex, worker, b, rep, m, n0=128, nf=1000)
