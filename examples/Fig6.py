from PUQ.performance import performanceModel
import matplotlib.pyplot as plt
from PUQ.performanceutils.utils import (
    plot_workers,
    plot_acc,
    plot_acqtime,
    plot_endtime,
    plot_errorend,
)

scale_list = [1, 1.1, 1.2]
acclevel = [0.1, 0.2, 0.25]
result = []
n = 1280

worker = 128
n0 = 0
batches = [1, 64, 128]
level = 0.1
for bid, b in enumerate(batches):
    PM = performanceModel(worker=worker, batch=b, n=n, n0=n0)
    PM.gen_gentime(1, 1, 0.25, typeGen="linear")
    PM.gen_simtime(1, 1, 0.1, typeSim="normal", seed=1)
    PM.gen_accuracy(-1, acclevel[bid], typeAcc="exponential")

    PM.simulate()
    PM.summarize()

    PM.complete(level)
    result.append(PM)

labs = ["$b=1$", "$b=64$", "$b=128$"]
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
plot_acc(axes[0], n, level, result, labellist=labs, logscale=False, fontsize=25, n0=n0)
plot_endtime(
    axes[1], n, level, result, labellist=labs, worker=worker, logscale=True, fontsize=25
)
plot_errorend(
    axes[2], n, level, result, labellist=labs, worker=worker, logscale=True, fontsize=25
)
plt.show()
