import numpy as np
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import (
    plot_acqtime,
    plot_endtime,
    plot_errorend,
)
import time

start = time.time()

repno = 1
n = 2560
varlist = [0.1, 10]
worker = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
simmeans = [0.1, 1, 10]
acqscale = [0.01, 0.1, 1]
accparams = [
    [-1, 0.2],
    [-1, 0.22],
    [-1, 0.24],
    [-1, 0.26],
    [-1, 0.28],
    [-1, 0.3],
    [-1, 0.32],
    [-1, 0.34],
    [-1, 0.36],
]
acclevel = 0.2
clist = ["b", "r", "g", "m", "y", "c"]
mlist = ["P", "o", "*", "s", "p", "h"]
linelist = ["-", "--", "-.", ":", "-.", ":"]
thrlist = []

lab = ["b=1", "b=2", "b=4", "b=8", "b=16", "b=32", "b=64", "b=128", "b=256"]
lw = 5
ms = 15
me = 200

result = []
for scaleid, scale in enumerate(acqscale):
    for varid, var in enumerate(varlist):
        for sid, sim_mean in enumerate(simmeans):
            res = []
            for r in range(repno):
                for id_b, b in enumerate(batches):
                    PM = performanceModel(worker=worker, batch=b, n=n, n0=worker)
                    PM.gen_acqtime(scale, scale, 0.001, typeGen="linear")
                    PM.gen_simtime(
                        sim_mean, sim_mean * var, 0.01, typeSim="normal", seed=r
                    )
                    PM.gen_curve(-1, accparams[id_b][1], typeAcc="exponential")
                    PM.simulate()
                    PM.summarize()
                    PM.complete(acclevel)

                    result.append(
                        {
                            "r": r,
                            "b": b,
                            "scale": scale,
                            "var": var,
                            "simmean": sim_mean,
                            "res": PM,
                        }
                    )

ft = 25
fig, axes = plt.subplots(2, len(acqscale), figsize=(24, 12))
for varid, var in enumerate(varlist):

    for scaleid, scale in enumerate(acqscale):

        res_c = [
            res for res in result if ((res["scale"] == scale) & (res["var"] == var))
        ]
        for sid, s in enumerate(simmeans):
            endtime = []

            for bid, b in enumerate(batches):
                endtime.append(
                    np.mean(
                        [
                            res["res"].complete_time
                            for res in res_c
                            if ((res["b"] == b) & (res["simmean"] == s))
                        ]
                    )
                )

            axes[varid, scaleid].plot(
                batches,
                endtime,
                marker=mlist[sid],
                markersize=ms,
                linestyle=linelist[sid],
                linewidth=lw,
                label=str(s),
                color=clist[sid],
            )
            axes[varid, scaleid].set_xscale("log")
            axes[varid, scaleid].set_yscale("log")
            axes[varid, scaleid].set_xticks(batches)
            axes[varid, scaleid].set_xticklabels(batches)
            axes[varid, scaleid].tick_params(
                axis="both", which="major", labelsize=ft - 5
            )
            axes[varid, scaleid].set_xlabel("b", fontsize=ft)
    axes[varid, 0].set_ylabel("Wall-clock time", fontsize=ft)
    if varid == len(varlist) - 1:
        handles, labels = axes[varid, 0].get_legend_handles_labels()

        labels = [r"$\tilde{a}$=" + l for l in labels]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            title_fontsize=ft,
            bbox_to_anchor=(0.5, 0.06),
            ncol=4,
            prop={"size": ft},
            fancybox=True,
            shadow=True,
        )
plt.savefig("Figure7.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()

end = time.time()
print("Elapsed time =", round(end - start, 3))
