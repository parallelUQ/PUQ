from plotutils import plotresult, plotparams
import matplotlib.pyplot as plt
import numpy as np


clist = ["b", "r", "g", "m", "y", "c"]
mlist = ["P", "o", "*", "s", "p", "h"]
linelist = ["-", "--", "-.", ":", "-.", ":"]


labelsb = ["b=1", "b=5", "b=25", "b=125"]
method = ["hybrid_ei"]
batch_sizes = [1, 5, 25, 125]
worker = 126
rep = 30
fonts = 25
nf = 1000
lw = 5
ms = 15
me = 200

path = "/Users/ozgesurer/Desktop/batch_mode/"
figs = [["sphere", "matyas", "ackley"], ["himmelblau", "holder", "easom"]]
for figno, example_name in enumerate(figs):
    for metric in ["AE"]:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        for exid, ex in enumerate(example_name):
            for bid, b in enumerate(batch_sizes):
                for mid, m in enumerate(method):
                    out = ex + "_" + m + "_" + "b" + str(b) + "_" + "w125"
                    avgAE, avgtime, avgTV = plotresult(
                        path, out, ex, worker, b, rep, m, n0=123, nf=nf
                    )
                    if metric == "AE":
                        axes[exid].plot(
                            np.arange(len(avgAE)),
                            avgAE,
                            label=labelsb[bid],
                            color=clist[bid],
                            linestyle=linelist[bid],
                            linewidth=lw,
                            marker=mlist[bid],
                            markersize=ms,
                            markevery=me,
                        )
                    else:
                        axes[exid].plot(
                            np.arange(len(avgTV)),
                            avgTV,
                            label=labelsb[bid],
                            color=clist[bid],
                            linestyle=linelist[bid],
                            linewidth=lw,
                            marker=mlist[bid],
                            markersize=ms,
                            markevery=me,
                        )
            axes[exid].set_yscale("log")
            axes[exid].set_xlabel("# of parameters", fontsize=fonts)
            if exid == 0:
                if metric == "AE":
                    axes[exid].set_ylabel(r"$\delta$", fontsize=fonts)
                else:
                    axes[exid].set_ylabel(r"MAD", fontsize=fonts)
            axes[exid].tick_params(axis="both", which="major", labelsize=fonts - 5)
        axes[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            shadow=True,
            ncol=4,
            fontsize=fonts,
        )
        if figno == 0:
            plt.savefig("Figure6.jpg", format="jpeg", bbox_inches="tight", dpi=500)
        else:
            plt.savefig("Figure14.jpg", format="jpeg", bbox_inches="tight", dpi=500)
        plt.show()
