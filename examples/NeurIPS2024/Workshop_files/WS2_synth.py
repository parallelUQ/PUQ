import numpy as np
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel
from mpl_toolkits.axes_grid1 import make_axes_locatable

repno = 1
n = 2560
varlist = [0.01, 1]
worker = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
s_mean = [2**0, 2**1, 2**2, 2**3, 2**4, 2**5]
a_mean = [2 ** (-5), 2 ** (-3), 2 ** (-1), 2**1, 2**3, 2**5]
sim_tick = [r"$2^0$", r"$2^1$", r"$2^2$", r"$2^3$", r"$2^4$", r"$2^5$"]
acq_tick = [r"$2^{-5}$", r"$2^{-3}$", r"$2^{-1}$", r"$2$", r"$2^3$", r"$2^5$"]

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


fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for vid, var in enumerate(varlist):
    result = []
    for aid, am in enumerate(a_mean):
        for sid, sm in enumerate(s_mean):
            res = []
            for r in range(repno):
                for id_b, b in enumerate(batches):
                    PM = performanceModel(worker=worker, batch=b, n=n, n0=worker)
                    PM.gen_acqtime(am, am, 2 ** (-5), typeGen="linear")
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

    timemat = np.zeros((len(a_mean), len(s_mean)))
    bmat = np.zeros((len(a_mean), len(s_mean)))
    for aid, am in enumerate(a_mean):
        for sid, sm in enumerate(s_mean):
            res_c = [res for res in result if ((res["am"] == am) & (res["sm"] == sm))]

            rmin = np.inf
            bmin = np.inf
            for bid, b in enumerate(batches):
                cs = [rs["res"].complete_time for rs in res_c if rs["b"] == b]
                if np.mean(cs) < rmin:
                    rmin = np.mean(cs)
                    bmin = b

            timemat[aid, sid] = rmin
            bmat[aid, sid] = bmin
    import matplotlib.colors as mcolors

    bounds = [
        0.5,
        1.5,
        3.5,
        7.5,
        15.5,
        31.5,
        63.5,
        127.5,
        255.5,
        512,
    ]  # bounds to differentiate colors
    cmap = plt.get_cmap("YlOrRd", len(batches))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalize colors based on bounds

    # Heatmap
    ft = 20
    divider = make_axes_locatable(ax[vid])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax[vid].imshow(bmat, cmap=cmap, norm=norm)

    # Show all ticks and label them with the respective list entries
    ax[vid].set_xticks(np.arange(len(s_mean)), labels=sim_tick)
    ax[vid].set_yticks(np.arange(len(a_mean)), labels=acq_tick)
    ax[vid].tick_params(axis="both", which="major", labelsize=ft - 5)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax[vid].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(a_mean)):
        for j in range(len(s_mean)):
            if bmat[i, j] >= 64:
                text = ax[vid].text(
                    j,
                    i,
                    int(timemat[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=ft - 5,
                )
            else:
                text = ax[vid].text(
                    j,
                    i,
                    int(timemat[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=ft - 5,
                )

    ax[vid].grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax[vid].set_xlabel("Simulation Time", fontsize=ft)
    ax[vid].set_ylabel("Acquisition Time", fontsize=ft)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=ft - 5)
    cbar.ax.set_ylabel("Batch Size", rotation=-90, va="bottom", fontsize=ft - 5)
    cbar.set_ticks(batches)
fig.suptitle("Variability in Simulation Time Increases \u2192", fontsize=ft)
fig.tight_layout()
plt.savefig("Figure_WS_2.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()
