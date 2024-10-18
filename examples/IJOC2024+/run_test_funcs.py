import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments
import scipy.stats as sps
from PUQ.prior import prior_dist
from test_funcs import holder, ackley, easom, sphere, matyas, himmelblau
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
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
    
    init_seeds = args.init_seeds
    final_seeds = args.final_seeds
    
    n_init = 10
    for s in np.arange(init_seeds, final_seeds):
    
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
                "mini_batch": 1,
                "nworkers": 2,
                "AL": args.al_func,
                "seed_n0": int(s),
                "prior": prior_func,
                "data_test": test_data,
                "max_evals": args.max_eval,
                "candsize": args.candsize,
                "refsize": args.refsize,
                "believer": args.believer,
            },
        )
    
        theta_al = al_data._info["theta"]
        ft = 20
        ms = 50
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap="RdGy")
        ax.scatter(theta_al[:, 0], theta_al[:, 1], c="black", marker="+", s=ms, zorder=2)
        ax.scatter(
            thetainit[:, 0],
            thetainit[:, 1],
            zorder=2,
            marker="o",
            facecolors="none",
            edgecolors="blue",
        )
        ax.set_xlabel(r"$\theta_1$", fontsize=ft)
        ax.set_ylabel(r"$\theta_2$", fontsize=ft)
        ax.tick_params(axis="both", labelsize=ft)
    
        plt.savefig(
            "Figure_" + args.funcname + ".jpg", format="jpeg", bbox_inches="tight", dpi=500
        )
        plt.show()
    
    end = time.time()
    print("Elapsed time =", round(end - start, 3))
