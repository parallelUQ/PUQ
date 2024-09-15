from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import (
    plot_acc,
    plot_acqtime,
    plot_endtime,
    plot_errorend,
)
import numpy as np
from result_read import get_rep_data
import matplotlib.pyplot as plt

n = 2048
w = 2
b = 1
rep = 30
example = "himmelblau_ex"
s = "himmelblau"
label = ["hybrid_ei_c1000/", "hybrid_ei_c100/", "hybrid_ei_c10/", "rnd/"]
labelf = ["hybrid_ei", "hybrid_ei", "hybrid_ei", "rnd"]
path = "/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/"

acclevel = 0.01
result = []
worker = 1
for mid, m in enumerate(label):

    PM = performanceModel(worker=1, batch=1, n=n, n0=0)

    # Read from existing experimental data
    filename = (
        path
        + "performanceAnalytics/new_fun_all/new_examples/"
        + example
        + "/"
        + label[mid]
    )
    avgae, avgtime = get_rep_data(s, w, b, rep, filename, labelf[mid])

    # Gen acq and sim time
    xt = np.arange(0, len(avgtime))
    xtest = np.arange(0, n)
    PM.gen_acqtime(xt, avgtime, xtest, typeGen="regress")
    PM.gen_simtime(0.0001, 0.0001, 0, typeSim="normal")

    # Fit a progress curve
    minl = np.min(avgae)
    maxl = np.max(avgae)
    lnew = [(litem - 0) / (maxl - 0) for litem in avgae]

    x_a = np.log(np.arange(1, len(lnew) + 1))
    y_a = np.log(lnew)
    xtest_a = np.log(np.arange(1, n + 1))
    PM.gen_curve(x_a, y_a, xtest_a, typeAcc="regress")
    PM.acc = np.exp(PM.acc)

    PM.simulate()
    PM.summarize()
    PM.complete(acclevel)
    result.append(PM)


lbl = [r"$\mathcal{A}_1$", r"$\mathcal{A}_2$", r"$\mathcal{A}_3$", r"$\mathcal{A}_4$"]
ft = 25
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
plot_acc(axes[0], n, acclevel, result, labellist=lbl, logscale=True, fontsize=ft, n0=1)
plot_endtime(
    axes[1],
    n,
    acclevel,
    result,
    labellist=lbl,
    worker=worker,
    logscale=True,
    fontsize=ft,
)
plot_errorend(
    axes[2],
    n,
    acclevel,
    result,
    labellist=lbl,
    worker=worker,
    logscale=True,
    fontsize=ft,
)
axes[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    fancybox=True,
    shadow=True,
    ncol=4,
    fontsize=ft,
)
plt.savefig("Figure1.jpg", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()
