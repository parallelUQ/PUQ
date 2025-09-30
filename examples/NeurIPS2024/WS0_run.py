import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
import scipy.stats as sps
from PUQ.prior import prior_dist
from test_funcs import holder, ackley, easom, sphere, matyas, himmelblau
import matplotlib.pyplot as plt
import time

start = time.time()

args = parse_arguments()

cls_data = eval(args.funcname)()

# # # Create a mesh for test set # # #
xpl = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 50)
ypl = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
setattr(cls_data, "theta", th.T)

ftest = np.zeros(2500)
for tid, t in enumerate(th.T):
    ftest[tid] = cls_data.function(t[0], t[1])
thetatest = th.T
ptest = np.zeros(thetatest.shape[0])
for i in range(ftest.shape[0]):
    mean = ftest[i]
    rnd = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
    ptest[i] = rnd.pdf(cls_data.real_data)

test_data = {"theta": thetatest, "f": ftest, "p": ptest, "p_prior": 1}

# # # # # # # # # # # # # # # # # # # # #
prior_func = prior_dist(dist="uniform")(
    a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]
)
# # # # # # # # # # # # # # # # # # # # #

init_seeds = 0  # args.init_seeds
final_seeds = 1  # args.final_seeds
args.max_eval = 512
n_init = 10
s = 0

for b in [32]:

    args.nworkers = 257
    args.minibatch = b

    thetainit = prior_func.rnd(n_init, s)
    finit = np.zeros(n_init)
    for tid, t in enumerate(thetainit):
        finit[tid] = cls_data.function(t[0], t[1])
    test_data["thetainit"] = thetainit
    test_data["finit"] = finit[None, :]

    al_data = designer(
        data_cls=cls_data,
        method="SEQCALOPT",
        args={
            "mini_batch": args.minibatch,
            "nworkers": args.minibatch + 1,
            "AL": "eivar",
            "seed_n0": int(s),
            "prior": prior_func,
            "data_test": test_data,
            "max_evals": args.max_eval,
            "candsize": args.candsize,
            "refsize": args.refsize,
            "believer": 2,
        },
    )
    save_output(
        al_data, cls_data.data_name, args.al_func, args.nworkers, args.minibatch, int(s)
    )

    plt.plot(al_data._info["TV"])
    # theta_al = al_data._info["theta"]
    # ft = 20
    # ms = 50
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap="RdGy")
    # ax.scatter(theta_al[:, 0], theta_al[:, 1], c="black", marker="+", s=ms, zorder=2)
    # ax.scatter(
    #     thetainit[:, 0],
    #     thetainit[:, 1],
    #     zorder=2,
    #     marker="o",
    #     facecolors="none",
    #     edgecolors="blue",
    # )
    # ax.set_xlabel(r"$\theta_1$", fontsize=ft)
    # ax.set_ylabel(r"$\theta_2$", fontsize=ft)
    # ax.tick_params(axis="both", labelsize=ft)
    # plt.show()
plt.yscale("log")
plt.show()
end = time.time()
print("Elapsed time =", round(end - start, 3))
