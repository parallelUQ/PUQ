import numpy as np
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel
import time
from matplotlib.ticker import FixedFormatter
from matplotlib.gridspec import GridSpec

start = time.time()

repno = 1
n = 2560
varlist = [0.1, 10]

worker = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
s_mean = [2**3, 2**6, 2**9]
a_mean = [2 ** (-2), 2 ** (-1), 2**0, 2**1, 2**2, 2**3, 2**4, 2**5]
a_tick = [
    r"$2^{-2}$",
    r"$2^{-1}$",
    r"$2^0$",
    r"$2^1$",
    r"$2^2$",
    r"$2^3$",
    r"$2^4$",
    r"$2^5$",
]
accparams = [
    [-1, 0.15],
    [-1, 0.18],
    [-1, 0.21],
    [-1, 0.24],
    [-1, 0.27],
    [-1, 0.30],
    [-1, 0.33],
    [-1, 0.36],
    [-1, 0.39],
]
acclevel = 0.2
ft = 25


fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05])

for vid, var in enumerate(varlist):
    for sid, sm in enumerate(s_mean):
        result = []
        for aid, am in enumerate(a_mean):
            for r in range(repno):
                for id_b, b in enumerate(batches):
                    PM = performanceModel(worker=worker, batch=b, n=n, n0=worker)
                    PM.gen_acqtime(am, am, 0.001, typeGen="linear")
                    PM.gen_simtime(sm, sm * var, 0.01, typeSim="normal", seed=r)
                    PM.gen_curve(-1, accparams[id_b][1], typeAcc="exponential")
                    PM.simulate()
                    PM.summarize()
                    PM.complete(acclevel)
                    result.append(
                        {
                            "r": r,
                            "b": b,
                            "am": am,
                            "var": var,
                            "sm": sm,
                            "res": PM,
                        }
                    )
                    print(PM.complete_no)

        timemat = np.zeros((len(a_mean), len(batches)))
        for aid, am in enumerate(a_mean):
            for bid, b in enumerate(batches):
                res_c = [res for res in result if ((res["am"] == am) & (res["b"] == b))]
                timemat[aid, bid] = res_c[0]["res"].complete_time

        subplot_ax = fig.add_subplot(gs[vid, sid])
        bo = np.argsort(np.argsort(timemat, axis=1), axis=1)
        im = subplot_ax.imshow(bo, aspect="auto", cmap="YlOrRd")
        if sid == 2:
            cbar_ax = fig.add_subplot(gs[vid, 3])
            cbar = fig.colorbar(
                im,
                cax=cbar_ax,
                ticks=np.array([0.0, 0.5, 1.0]) * bo.max(),
                format=FixedFormatter(["lowest", "middle", "highest"]),
            )
            cbar.ax.tick_params(labelsize=ft - 5)
            cbar.ax.set_ylabel(
                "Wall-clock time", rotation=-90, va="bottom", fontsize=ft
            )

        # # Show all ticks and label them with the respective list entries
        subplot_ax.set_xticks(np.arange(len(batches)), labels=batches)
        subplot_ax.set_yticks(np.arange(len(a_mean)), labels=a_tick)

        # Rotate the tick labels and set their alignment.
        plt.setp(
            subplot_ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        if vid == 1:
            subplot_ax.set_xlabel(r"Batch size ($b$)", fontsize=ft)
        if sid == 0:
            subplot_ax.set_ylabel(r"Acquisition time ($\tilde{a}^A$)", fontsize=ft)

        subplot_ax.tick_params(axis="both", which="major", labelsize=ft - 5)

        if vid == 0:
            if sid == 0:
                subplot_ax.set_title(r"$\tilde{a}^S = 2^3$", fontsize=ft)
            elif sid == 1:
                subplot_ax.set_title(r"$\tilde{a}^S = 2^6$", fontsize=ft)
            elif sid == 2:
                subplot_ax.set_title(r"$\tilde{a}^S = 2^9$", fontsize=ft)

        # for aid, am in enumerate(a_mean):
        #     for bid, b in enumerate(batches):
        #         text = ax[vid, sid].text(bid, aid, np.round(timemat[aid, bid]/np.min(timemat), 1),
        #                                ha="center", va="center", color="black", fontsize=ft-10)


fig.suptitle("Simulation time increases \u2192", fontsize=ft)
plt.savefig("Figure8.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()


end = time.time()
print("Elapsed time =", round(end - start, 3))
