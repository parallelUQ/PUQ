import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PUQ.performance import performanceModel
import matplotlib.patches as patches

# Create a figure
fig = plt.figure(figsize=(24, 6))

# Define a GridSpec layout
gs = gridspec.GridSpec(1, 3)  # 1 row, 3 columns

# Create subplots with varying spaces between them by adjusting the positions
ax1 = fig.add_subplot(gs[0])  # First subplot
ax2 = fig.add_subplot(gs[1])  # Second subplot
ax3 = fig.add_subplot(gs[2])  # Third subplot

# Adjust positions manually for different spacing
ax1.set_position([0.1, 0.1, 0.25, 0.8])  # [left, bottom, width, height]
ax2.set_position([0.43, 0.1, 0.25, 0.8])  # Less width, more space on the left
ax3.set_position([0.76, 0.1, 0.25, 0.8])  # Normal width, more space between ax2 and ax3

# Plot data on the subplots
acclevel = [0.1, 0.2, 0.25]
n = 1280
ft = 25
worker = 128
n0 = 0
batches = [1, 64, 128]
clist = ["b", "r", "g", "m", "y", "c"]
mlist = ["P", "o", "*", "s", "p", "h"]
linelist = ["-", "--", "-.", ":", "-.", ":"]
level = 0.3
for bid, b in enumerate(batches):
    PM = performanceModel(worker=worker, batch=b, n=n, n0=n0)
    PM.gen_curve(-1, acclevel[bid], typeAcc="exponential")
    PM.gen_acqtime(1, 1, 0.25, typeGen="linear")
    PM.gen_simtime(1, 1, 0.1, typeSim="normal", seed=1)
    PM.simulate()
    PM.summarize()

    PM.complete(level)

    ax1.plot(PM.acc[0:500], linestyle=linelist[bid], linewidth=5.0, color=clist[bid])

    ax1.hlines(y=level, xmin=0, xmax=500, linewidth=5, color="k")

    ax1.vlines(
        x=PM.complete_no,
        ymin=0,
        ymax=level,
        linewidth=5,
        color=clist[bid],
        linestyles=(0, (2, 5)),
    )

# Annotating outside of ax1 using figure-relative coordinates
ax1.annotate(
    r"$\alpha$",
    xy=(0.3, 0.34),
    xycoords="figure fraction",
    fontsize=ft,
    fontweight="bold",
    color="black",
)

# # Define arrow properties
# arrow = patches.FancyArrowPatch(
#     (0.35, 0.35),  # Start point in figure coordinates
#     (0.38, 0.35),  # End point in figure coordinates
#     mutation_scale=50,  # Size of the arrow
#     arrowstyle="->",  # Arrow style
#     color="black",
#     linewidth=5,
#     transform=fig.transFigure,  # Use figure coordinates
# )

# # Add the arrow to the figure
# fig.patches.append(arrow)

ax1.set_xlabel("# of parameters", fontsize=ft)
ax1.set_ylabel("Error", fontsize=ft)
ax1.tick_params(axis="both", which="major", labelsize=ft - 5)
ax1.set_xlim(0, 500)
ax1.set_ylim(0, 1.1)

# AXIS 2
# Generate random data
xmax = 25
np.random.seed(1)
means = [5, 10, 15, 20]

datas = []
for m in means:
    rnddat = np.random.normal(m, 2, 30)
    rnddat = [0.1 if r < 0 else r for r in rnddat]
    datas.append(rnddat)

# Set the number of bins
bins = np.linspace(-2, xmax, 30)

# Plot histograms horizontally with vertical shifts
for i, data in enumerate(datas):
    counts, _ = np.histogram(data, bins=bins)
    # Plot histogram as a horizontal bar plot
    ax2.barh(
        bins[:-1],
        counts,
        height=0.8,
        color="red",
        alpha=0.7,
        label=i,
        edgecolor="none",
        align="center",
        left=i * 10,
    )

# Add labels and title
ax2.set_xlabel("t", fontsize=ft)
ax2.set_ylabel("Acquisition time", fontsize=ft)

# Set custom x-axis ticks and labels
tick_positions = [0, 10, 20, 30]
tick_labels = ["0", "1", "2", "3"]

ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)
ax2.plot([0, 10, 20, 30], [5, 10, 15, 20], color="black", linestyle="--", linewidth=5)
ax2.scatter([0, 10, 20, 30], [5, 10, 15, 20], color="black", marker="*", s=1000)
ax2.tick_params(axis="both", which="major", labelsize=ft - 5)
ax2.set_ylim(0, 25)

# AXIS 3
ax3.hist(
    np.random.normal(5, 2, 100), bins=30, edgecolor="white", color="blue", alpha=0.5
)
ax3.set_xlabel("Simulation time", fontsize=ft)
ax3.set_ylabel("Frequency", fontsize=ft)
ax3.tick_params(axis="both", which="major", labelsize=ft - 5)
# ax3.vlines(x=5, ymin=0, ymax=9, color="black", linestyle="--", linewidth=5)

# Set titles to distinguish the plots
ax1.set_title(r"Find $n_k(b, \alpha)$ for worker size $w$", fontsize=ft)
ax2.set_title("Histogram of $a_{\omega,k}(b,t)$", fontsize=ft)
ax3.set_title("Histogram of $s_{\omega,j}$", fontsize=ft)
ax3.set_xlim(0, 11)
# plt.annotate(
#     "1",
#     (0.05, 1),
#     xycoords="figure fraction",
#     bbox={"boxstyle": "circle", "color": "lightgrey"},
#     fontsize=ft,
#     color="black",
# )

# plt.annotate(
#     "2",
#     (0.4, 1),
#     xycoords="figure fraction",
#     bbox={"boxstyle": "circle", "color": "lightgrey"},
#     fontsize=ft,
#     color="black",
# )

# plt.annotate(
#     "3",
#     (0.7, 1),
#     xycoords="figure fraction",
#     bbox={"boxstyle": "circle", "color": "lightgrey"},
#     fontsize=ft,
#     color="black",
# )

plt.gca().set_aspect("equal", adjustable="box")
plt.savefig("Figure3.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()
