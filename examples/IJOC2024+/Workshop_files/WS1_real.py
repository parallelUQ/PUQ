import numpy as np
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel
from matplotlib.ticker import FixedFormatter
from plotutils import plotresult


repno = 5
n = 2560
worker = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
s_mean = [2, 2**6]
a_mean = [
    2 ** (-2),
    2 ** (-1),
    2 ** (0),
    2 ** (1),
    2 ** (2),
    2 ** (3),
    2 ** (4),
    2 ** (5),
]
acq_tick = [
    r"$2^{-2}$",
    r"$2^{-1}$",
    r"$2^{0}$",
    r"$2^1$",
    r"$2^2$",
    r"$2^3$",
    r"$2^4$",
    r"$2^5$",
]

acclevel = 1.7536062885144855e-05
rep = 0
ex = "himmelblau"
m = "eivar"
path = "/Users/ozgesurer/Desktop/WS_data/batch_newjobs/"

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for sid, sm in enumerate(s_mean):
    result = []
    for aid, am in enumerate(a_mean):
        for r in range(repno):
            for id_b, b in enumerate(batches):
                PM = performanceModel(worker=worker, batch=b, n=n, n0=worker)
                PM.gen_acqtime(am, am, 0, typeGen="linear")
                PM.gen_simtime(sm, sm, 0.01, typeSim="normal", seed=r)

                out = "b" + str(b)
                avgAE, avgtime, avgTV = plotresult(
                    path, out, ex, b + 1, b, rep, m, n0=0, nf=n + 1
                )
                PM.acc = avgTV

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
            timemat[aid, bid] = np.mean(
                [r["res"].complete_time for r in res_c]
            )  # res_c[0]["res"].complete_time

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
