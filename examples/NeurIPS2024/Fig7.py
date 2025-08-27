from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import (
    plot_acc,
    plot_endtime,
    plot_errorend,
)
import matplotlib.pyplot as plt
import time

start = time.time()
scale_list = [1, 1.1, 1.2]
acclevel = [0.1, 0.2, 0.25]
result = []
n = 1280
ft = 25
worker = 128
n0 = 0
batches = [1, 64, 128]
level = 0.1
for bid, b in enumerate(batches):
    PM = performanceModel(worker=worker, batch=b, n=n, n0=n0)
    PM.gen_acqtime(1, 1, 0.25, typeGen="linear")
    PM.gen_simtime(1, 1, 0.1, typeSim="normal", seed=1)
    PM.gen_curve(-1, acclevel[bid], typeAcc="exponential")

    PM.simulate()
    # PM.summarize()

    PM.complete(level)
    result.append(PM)

labs = ["$b=1$", "$b=64$", "$b=128$"]
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
plot_acc(axes[0], n, level, result, labellist=labs, logscale=False, fontsize=ft, n0=n0)
plot_endtime(
    axes[1], n, level, result, labellist=labs, worker=worker, logscale=True, fontsize=ft
)
plot_errorend(
    axes[2], n, level, result, labellist=labs, worker=worker, logscale=True, fontsize=ft
)
axes[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    fancybox=True,
    shadow=True,
    ncol=4,
    fontsize=ft,
)
plt.savefig("Figure7.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()

end = time.time()
print("Elapsed time =", round(end - start, 3))
