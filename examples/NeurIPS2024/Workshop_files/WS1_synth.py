import numpy as np
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FixedFormatter


repno = 1
n = 2560
varlist = [0.1, 10]
worker = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
s_mean = [2, 2**6]
a_mean = [2 ** (-2), 2 ** (-1), 2**0, 2**1, 2**2, 2**3, 2**4, 2**5]
acq_tick = [
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


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for sid, sm in enumerate(s_mean):
    result = []
    for aid, am in enumerate(a_mean):
        for r in range(repno):
            for id_b, b in enumerate(batches):
                PM = performanceModel(worker=worker, batch=b, n=n, n0=worker)
                PM.gen_acqtime(am, am, 0.001, typeGen="linear")
                PM.gen_simtime(sm, sm * 1, 0.01, typeSim="normal", seed=r)

                PM.gen_curve(-1, accparams[id_b][1], typeAcc="exponential")

                PM.simulate()
                PM.summarize()
                PM.complete(acclevel)

                result.append(
                    {
                        "r": r,
                        "b": b,
                        "am": am,
                        "var": 1,
                        "sm": sm,
                        "res": PM,
                    }
                )

    timemat = np.zeros((len(a_mean), len(batches)))
    bmat = np.zeros((len(a_mean), len(batches)))
    for aid, am in enumerate(a_mean):
        for bid, b in enumerate(batches):
            res_c = [res for res in result if ((res["am"] == am) & (res["b"] == b))]
            timemat[aid, bid] = res_c[0]["res"].complete_time

    ft = 20
    bo = np.argsort(np.argsort(timemat, axis=1), axis=1)
    im = ax[sid].imshow(bo, aspect="auto", cmap="YlOrRd")
    cbar = fig.colorbar(
        im,
        ticks=np.array([0.0, 0.5, 1.0]) * bo.max(),
        format=FixedFormatter(["low", "mid", "high"]),
    )
    cbar.ax.tick_params(labelsize=ft - 5)
    cbar.ax.set_ylabel("Wall-Clock Time", rotation=-90, va="bottom", fontsize=ft - 5)
    # # Show all ticks and label them with the respective list entries
    ax[sid].set_xticks(np.arange(len(batches)), labels=batches)
    ax[sid].set_yticks(np.arange(len(a_mean)), labels=acq_tick)
    ax[sid].set_xlabel("Batch Size", fontsize=ft)
    ax[sid].set_ylabel("Acquisition Time", fontsize=ft)
    ax[sid].tick_params(axis="both", which="major", labelsize=ft - 5)
fig.suptitle("Simulation Time Increases \u2192", fontsize=ft)
fig.tight_layout()
plt.savefig("Figure_WS_1.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()
