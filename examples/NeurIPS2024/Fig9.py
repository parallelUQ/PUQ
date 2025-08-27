import numpy as np
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

repno = 10
n = 2560
worker = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
s_mean = [2**0, 2**1, 2**2, 2**3, 2**4, 2**5]
a_mean = [2 ** (-3), 2 ** (-2), 2 ** (-1), 2**0, 2**1, 2**2]
sim_tick = [r"$2^0$", r"$2^1$", r"$2^2$", r"$2^3$", r"$2^4$", r"$2^5$"]
acq_tick = [r"$2^{-3}$", r"$2^{-2}$", r"$2^{-1}$", r"$2^{0}$", r"$2^1$", r"$2^2$"]
acclevel = 0.2
accparams = [[-1, 0.1], [-1, 0.3], [-1, 0.5]]

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

for accid, acc in enumerate(accparams):
    result = []
    for aid, am in enumerate(a_mean):
        for sid, sm in enumerate(s_mean):
            for r in range(repno):
                for id_b, b in enumerate(batches):
                    PM = performanceModel(worker=worker, batch=b, n=n, n0=worker)
                    PM.gen_acqtime(am, am, 2 ** (-5), typeGen="linear")
                    PM.gen_simtime(sm, sm, 0.01, typeSim="normal", seed=r)
                    PM.gen_curve(acc[0], acc[1] + id_b * 0.02, typeAcc="exponential")
                    PM.simulate()
                    # PM.summarize()
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
                    # print(PM.complete_no)

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

    ft = 25
    subplot_ax = fig.add_subplot(gs[accid])
    # Show all ticks and label them with the respective list entries
    subplot_ax.set_xticks(np.arange(len(s_mean)), labels=sim_tick)
    subplot_ax.set_yticks(np.arange(len(a_mean)), labels=acq_tick)
    subplot_ax.tick_params(axis="both", which="major", labelsize=ft - 5)
    # Rotate the tick labels and set their alignment.
    plt.setp(
        subplot_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

    # Loop over data dimensions and create text annotations.
    for i in range(len(a_mean)):
        for j in range(len(s_mean)):
            if bmat[i, j] >= 64:
                text = subplot_ax.text(
                    j,
                    i,
                    int(timemat[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=ft - 5,
                )
            else:
                text = subplot_ax.text(
                    j,
                    i,
                    int(timemat[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=ft - 5,
                )

    subplot_ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    subplot_ax.set_xlabel(r"Simulation time ($\breve{a}^S$)", fontsize=ft)
    if accid == 0:
        subplot_ax.set_ylabel(r"Acquisition time ($\breve{a}^A$)", fontsize=ft)

    # bounds to differentiate colors
    bounds = [0.5, 1.5, 3.5, 7.5, 15.5, 31.5, 63.5, 127.5, 255.5, 512]
    cmap = plt.get_cmap("YlOrRd", len(batches))
    # Normalize colors based on bounds
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im = subplot_ax.imshow(bmat, cmap=cmap, norm=norm)

    if accid == 2:
        cbar_ax = fig.add_subplot(gs[3])

        # Heatmap
        # divider = make_axes_locatable(cbar_ax)
        # cax = divider.append_axes('right', size='5%', pad=0.2)
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
        cbar_ax.set_aspect(20)
        cbar.ax.tick_params(labelsize=ft - 5)
        cbar.set_ticks(batches)
        cbar.ax.set_ylabel(r"Batch size ($b$)", rotation=-90, va="bottom", fontsize=ft)

    if accid == 0:
        subplot_ax.set_title(r"$\mathcal{A}_1$", fontsize=ft)
    elif accid == 1:
        subplot_ax.set_title(r"$\mathcal{A}_2$", fontsize=ft)
    elif accid == 2:
        subplot_ax.set_title(r"$\mathcal{A}_3$", fontsize=ft)

plt.savefig("Figure9.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()
