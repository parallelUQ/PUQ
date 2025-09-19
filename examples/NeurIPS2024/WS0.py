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

labelsb = ["EIVAR"]

batch = 1
worker = 2
rep = 0
fonts = 25

path = "/Users/ozgesurer/Desktop/WS_data/batch_newjobs/"
ex = "himmelblau"
m = "eivar"
markers = ["o", "+", "*", "P", "^", "D", "s", "p", "h"]
colors = [
    "red",
    "blue",
    "orange",
    "magenta",
    "cyan",
    "green",
    "purple",
    "yellow",
    "pink",
]

n = 2560
ft = 20
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for id_b, b in enumerate([1, 2, 4, 8, 16, 32, 64, 128, 256]):

    out = "b" + str(b)
    avgAE, avgtime, avgTV = plotresult(path, out, ex, b + 1, b, rep, m, n0=0, nf=n + 1)

    # if b == 256:
    # print(avgTV[-1])

    print(np.where(avgTV <= 1.7536062885144855e-05)[0][0])

    axes[0].plot(np.arange(0, n + 1), avgTV, c=colors[id_b])

    axes[0].plot(
        np.arange(0, n + 1),
        avgTV,
        markers[id_b],
        markevery=256,
        markersize=ms,
        label=str(b),
        c=colors[id_b],
    )

    axes[1].plot(
        np.arange(0, len(np.unique(avgTV))),
        sorted(np.unique(avgTV), reverse=True),
        c=colors[id_b],
    )

    axes[1].plot(
        np.arange(0, len(np.unique(avgTV))),
        sorted(np.unique(avgTV), reverse=True),
        markers[id_b],
        markersize=ms,
        label=str(b),
        c=colors[id_b],
    )

    print(sum(avgtime))

    axes[2].scatter(
        np.arange(0, len(avgtime[0:][avgtime[0:] > 0])),
        avgtime[0:][avgtime[0:] > 0],
        marker=markers[id_b],
        s=ms * 2,
        label=str(b),
        c=colors[id_b],
    )


axes[0].set_xlabel("# of sim evals", fontsize=ft)
axes[0].set_ylabel("MAD", fontsize=ft)
axes[1].set_xlabel("# of stages", fontsize=ft)
axes[1].set_ylabel("MAD", fontsize=ft)
axes[2].set_xlabel("# of stages", fontsize=ft)
axes[2].set_ylabel("Acquisition time", fontsize=ft)

axes[1].set_xscale("log")
axes[0].set_yscale("log")
axes[1].set_yscale("log")
axes[2].set_yscale("log")
axes[2].set_xscale("log")

axes[2].tick_params(axis="both", which="major", labelsize=ft - 10)
axes[0].tick_params(axis="both", which="major", labelsize=ft - 10)
axes[1].tick_params(axis="x", which="major", labelsize=ft - 10)
axes[1].tick_params(axis="y", which="major", length=0)


axes[0].legend(
    loc="upper center", bbox_to_anchor=(1.6, -0.25), ncol=9, fontsize=fonts - 10
)
# fig.tight_layout()
plt.savefig("Figure_WS_0.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()
