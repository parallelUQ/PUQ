import numpy as np
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel
import time
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

start = time.time()

repno = 10
varlist = [1]
acclevel = 0.2
n = 2560
workers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
s_mean = [2**0, 2**3, 2**6]
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
am = 2 ** (-1)

ft = 25
lw = 5
ms = 15
me = 200

fig = plt.figure(figsize=(20, 15))
gs = GridSpec(3, 6, width_ratios=[1, 0.04, 1, 0.04, 1, 0.04])

for sid, sm in enumerate(s_mean):
    result = []
    for id_b, b in enumerate(batches):
        for r in range(repno):
            for id_w, w in enumerate(workers):
                if b <= w:
                    PM = performanceModel(worker=w, batch=b, n=n, n0=w)
                    PM.gen_acqtime(am, am, 2 ** (-6), typeGen="linear")
                    PM.gen_simtime(sm, sm, 0.001, typeSim="normal", seed=r)
                    PM.gen_curve(
                        accparams[id_b][0], accparams[id_b][1], typeAcc="exponential"
                    )
                    PM.simulate()
                    # PM.summarize()
                    PM.complete(acclevel)
                    result.append(
                        {
                            "r": r,
                            "b": b,
                            "var": 1,
                            "w": w,
                            "sm": sm,
                            "res": PM,
                        }
                    )

    idle = np.zeros((len(batches), len(workers)))
    computing = np.zeros((len(batches), len(workers)))
    endtime = np.zeros((len(batches), len(workers)))
    endtime_temp = np.zeros((len(batches), len(workers)))
    for wid, w in enumerate(workers):
        for bid, b in enumerate(batches):
            if b <= w:
                res_c = [res for res in result if ((res["w"] == w) & (res["b"] == b))]
                idle[bid, wid] = np.mean(
                    [res_c[i]["res"].avg_idle_time for i in range(0, repno)]
                )
                computing[bid, wid] = np.mean(
                    [res_c[i]["res"].computing_hours for i in range(0, repno)]
                )
                endtime[bid, wid] = np.mean(
                    [res_c[i]["res"].complete_time for i in range(0, repno)]
                )
                endtime_temp[bid, wid] = np.mean(
                    [res_c[i]["res"].complete_time for i in range(0, repno)]
                )
            else:
                idle[bid, wid] = np.nan
                computing[bid, wid] = np.nan
                endtime[bid, wid] = np.nan
                endtime_temp[bid, wid] = np.nan

    for bid, b in enumerate(batches):
        for wid, w in enumerate(workers):
            if b <= w:
                endtime[bid, wid] = endtime[bid, wid] / endtime_temp[bid, bid]

    bidle, bcomph, bend = (
        np.zeros(len(workers)),
        np.zeros(len(workers)),
        np.zeros(len(workers)),
    )
    for wid, w in enumerate(workers):
        bidle[wid] = np.nanargmin(idle[:, wid])
        bcomph[wid] = np.nanargmin(computing[:, wid])
        bend[wid] = np.nanargmin(endtime_temp[:, wid])

    widle, wcomph, wend = (
        np.zeros(len(workers)),
        np.zeros(len(workers)),
        np.zeros(len(workers)),
    )
    for bid, b in enumerate(batches):
        widle[bid] = np.nanargmin(idle[bid, :])
        wcomph[bid] = np.nanargmin(computing[bid, :])
        wend[bid] = np.nanargmin(endtime_temp[bid, :])

    if sid == 0:
        colid = 0
    elif sid == 1:
        colid = 2
    elif sid == 2:
        colid = 4

    # AXIS 1
    subplot_ax = fig.add_subplot(gs[1, colid])
    masked_data = np.ma.masked_invalid(idle)
    norm = mcolors.LogNorm(vmin=np.nanmin(idle), vmax=np.nanmax(idle))
    im = subplot_ax.imshow(masked_data, norm=norm, cmap="YlOrRd")

    cbar_ax = fig.add_subplot(gs[1, colid + 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=ft - 8)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_tick_params(width=2, color="black")

    if sid == 2:
        cbar.ax.set_ylabel("Idle time", rotation=-90, va="bottom", fontsize=ft)

    # # Show all ticks and label them with the respective list entries
    subplot_ax.set_yticks(np.arange(len(batches)), labels=batches)
    subplot_ax.set_xticks(np.arange(len(workers)), labels=workers)
    subplot_ax.tick_params(axis="both", which="major", labelsize=ft - 5)
    plt.setp(
        subplot_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    if sid == 0:
        subplot_ax.set_ylabel(r"Batch size ($b$)", fontsize=ft)

    # AXIS 2
    subplot_ax = fig.add_subplot(gs[2, colid])
    masked_data = np.ma.masked_invalid(computing)
    norm = mcolors.LogNorm(vmin=np.nanmin(computing), vmax=np.nanmax(computing))
    im = subplot_ax.imshow(masked_data, norm=norm, cmap="YlOrRd")
    cbar_ax = fig.add_subplot(gs[2, colid + 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=ft - 5)
    if sid == 2:
        cbar.ax.set_ylabel("Computing hours", rotation=-90, va="bottom", fontsize=ft)
    # Adjust ticks to be on the left side
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_tick_params(width=2, color="black")
    # # Show all ticks and label them with the respective list entries
    subplot_ax.set_yticks(np.arange(len(batches)), labels=batches)
    subplot_ax.set_xticks(np.arange(len(workers)), labels=workers)
    if sid == 0:
        subplot_ax.set_ylabel(r"Batch size ($b$)", fontsize=ft)
    subplot_ax.tick_params(axis="both", which="major", labelsize=ft - 5)
    subplot_ax.set_xlabel(r"# of workers ($w$)", fontsize=ft)
    plt.setp(
        subplot_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    # AXIS 0
    subplot_ax = fig.add_subplot(gs[0, colid])
    masked_data = np.ma.masked_invalid(endtime)

    bounds = [
        0.0,
        2 ** (-9),
        2 ** (-8),
        2 ** (-7),
        2 ** (-6),
        2 ** (-5),
        2 ** (-4),
        2 ** (-3),
        2 ** (-2),
        2 ** (-1),
        2**0,
    ]  # bounds to differentiate colors
    cmap = plt.get_cmap("YlOrRd", 11)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalize colors based on bounds

    cbar_ax = fig.add_subplot(gs[0, colid + 1])
    im = subplot_ax.imshow(masked_data, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=ft - 5)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_tick_params(width=2, color="black")
    cbar.set_ticks(
        [
            0.0,
            2 ** (-9),
            2 ** (-8),
            2 ** (-7),
            2 ** (-6),
            2 ** (-5),
            2 ** (-4),
            2 ** (-3),
            2 ** (-2),
            2 ** (-1),
            2**0,
        ]
    )
    cbar.set_ticklabels(
        [
            r"$0$",
            r"$2^{-9}$",
            r"$2^{-8}$",
            r"$2^{-7}$",
            r"$2^{-6}$",
            r"$2^{-5}$",
            r"$2^{-4}$",
            r"$2^{-3}$",
            r"$2^{-2}$",
            r"$2^{-1}$",
            r"$2^{0}$",
        ]
    )
    if sid == 2:
        cbar.ax.set_ylabel(
            "Relative wall-clock time", rotation=-90, va="bottom", fontsize=ft
        )

    # # Show all ticks and label them with the respective list entries
    subplot_ax.set_yticks(np.arange(len(batches)), labels=batches)
    subplot_ax.set_xticks(np.arange(len(workers)), labels=workers)
    if sid == 0:
        subplot_ax.set_ylabel(r"Batch size ($b$)", fontsize=ft)
    subplot_ax.tick_params(axis="both", which="major", labelsize=ft - 5)
    plt.setp(
        subplot_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

    if sid == 0:
        if colid == 0:
            subplot_ax.set_title(r"$\breve{a}^S = 2^0$", fontsize=ft)
    if sid == 1:
        if colid == 2:
            subplot_ax.set_title(r"$\breve{a}^S = 2^3$", fontsize=ft)
    if sid == 2:
        if colid == 4:
            subplot_ax.set_title(r"$\breve{a}^S = 2^6$", fontsize=ft)
    # for i in range(len(batches)):
    #     for j in range(len(workers)):
    #         if batches[i] <= workers[j]:
    #             text = ax[0, sid].text(j, i, np.round(endtime[i, j], 1),
    #                                    ha="center", va="center", color="black", fontsize=ft-10)


fig.suptitle("Simulation time increases \u2192", fontsize=ft)
fig.subplots_adjust(top=0.9, bottom=0.1)
# fig.tight_layout()
plt.savefig("Figure10.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()
