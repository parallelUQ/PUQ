import numpy as np
from result_read import get_rep_data
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import (
    find_threshold,
    plot_workers,
    plot_acc,
    plot_acqtime,
    plot_endtime,
    plot_errorend,
)

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
mlist = ["P", "p", "*", "o", "s", "h"]
linelist = ["-", "--", "-.", ":", "-.", ":"]
thrlist = []

lab = ["b=1", "b=2", "b=4", "b=8", "b=16", "b=32", "b=64", "b=128", "b=256"]


result = []
for scaleid, scale in enumerate(acqscale):
    for varid, var in enumerate(varlist):
        for sid, sim_mean in enumerate(simmeans):
            res = []
            for r in range(repno):
                for id_b, b in enumerate(batches):
                    PM = performanceModel(worker=worker, batch=b, n=n, n0=worker)
                    PM.gen_gentime(scale, scale, 0.001, typeGen="linear")
                    PM.gen_simtime(
                        sim_mean, sim_mean * var, 0.01, typeSim="normal", seed=r
                    )
                    PM.gen_accuracy(-1, accparams[id_b][1], typeAcc="exponential")
                    PM.simulate()
                    PM.summarize()
                    PM.complete(acclevel)

                    # plot_workers(PM, PM.job_list, PM.stage_list)
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

ft = 18
for varid, var in enumerate(varlist):
    fig, axes = plt.subplots(1, len(acqscale), figsize=(24, 6))
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

            axes[scaleid].plot(
                batches,
                endtime,
                marker=mlist[sid],
                markersize=10,
                linestyle=linelist[sid],
                linewidth=2.0,
                label=str(s),
                color=clist[sid],
            )
            axes[scaleid].set_xscale("log")
            axes[scaleid].set_yscale("log")
            axes[scaleid].set_xticks(batches)
            axes[scaleid].set_xticklabels(batches, fontsize=14)
            axes[scaleid].tick_params(axis="both", which="major", labelsize=ft)
            axes[scaleid].set_xlabel("b", fontsize=ft)
    axes[0].set_ylabel("Wall-clock time", fontsize=ft + 2)
    if varid == len(varlist) - 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            title_fontsize=25,
            title=r"$\tilde{a}$",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            prop={"size": 18},
            fancybox=True,
            shadow=True,
        )
    plt.show()
