from plotutils import plotresult, plotparams
import matplotlib.pyplot as plt
import numpy as np

# Generates Fig 3, 4, 11, 12
clist = ["b", "r", "g", "m", "y", "c"]
mlist = ["P", "o", "*", "s", "p", "h"]
linelist = ["-", "--", "-.", ":", "-.", ":"]

lw = 5
ms = 15
me = 200

labelsb = ["EI", "EIVAR", "HYBRID", "RND"]
method = ["ei", "eivar", "hybrid_ei", "rnd"]
batch = 1
worker = 2
rep = 30
fonts = 25

path = "/Users/ozgesurer/Desktop/sh_files/"
figs = [["himmelblau", "holder", "easom"], ["sphere", "matyas", "ackley"]]
for ex_id, example_name in enumerate(figs):
    for metric in ["AE", "MAD"]:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        for exid, ex in enumerate(example_name):
            for mid, m in enumerate(method):
                out = ex + "_" + m
                avgAE, avgtime, avgTV = plotresult(
                    path, out, ex, worker, batch, rep, m, n0=0, nf=1000
                )
                # print(avgAE[0:10])
                if metric == "AE":
                    axes[exid].plot(
                        np.arange(len(avgAE)),
                        avgAE,
                        label=labelsb[mid],
                        color=clist[mid],
                        linestyle=linelist[mid],
                        linewidth=lw,
                        marker=mlist[mid],
                        markersize=ms,
                        markevery=me,
                    )
                else:
                    axes[exid].plot(
                        np.arange(len(avgTV)),
                        avgTV,
                        label=labelsb[mid],
                        color=clist[mid],
                        linestyle=linelist[mid],
                        linewidth=lw,
                        marker=mlist[mid],
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
        if ex_id == 0:
            if metric == "AE":
                plt.savefig("Figure4.jpg", format="jpeg", bbox_inches="tight", dpi=500)
            else:
                plt.savefig("Figure5.jpg", format="jpeg", bbox_inches="tight", dpi=500)
        else:
            if metric == "AE":
                plt.savefig("Figure12.jpg", format="jpeg", bbox_inches="tight", dpi=500)
            else:
                plt.savefig("Figure13.jpg", format="jpeg", bbox_inches="tight", dpi=500)

        plt.show()
