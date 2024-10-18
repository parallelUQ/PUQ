import numpy as np
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments
import scipy.stats as sps
from PUQ.prior import prior_dist
from test_funcs import himmelblau
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    start = time.time()
    
    args = parse_arguments()
    
    example = "himmelblau"
    cls_data = eval(example)()
    
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
    
    init_seeds = 2
    final_seeds = 3
    
    n_init = 10
    for s in np.arange(init_seeds, final_seeds):
    
        thetainit = prior_func.rnd(n_init, s)
        finit = np.zeros(n_init)
        for tid, t in enumerate(thetainit):
            finit[tid] = cls_data.function(t[0], t[1])
        test_data["thetainit"] = thetainit
        test_data["finit"] = finit[None, :]
    
        al_data_ei = designer(
            data_cls=cls_data,
            method="SEQCALOPT",
            args={
                "mini_batch": 1,
                "nworkers": 2,
                "AL": "ei",
                "seed_n0": int(s),
                "prior": prior_func,
                "data_test": test_data,
                "max_evals": 200,
                "candsize": args.candsize,
                "refsize": args.refsize,
                "believer": args.believer,
            },
        )
    
        al_data_hyb = designer(
            data_cls=cls_data,
            method="SEQCALOPT",
            args={
                "mini_batch": 1,
                "nworkers": 2,
                "AL": "hybrid_ei",
                "seed_n0": int(s),
                "prior": prior_func,
                "data_test": test_data,
                "max_evals": 200,
                "candsize": args.candsize,
                "refsize": args.refsize,
                "believer": args.believer,
            },
        )
    
        show = True
        ft = 20
        ms = 50
        if show:
            theta_ei = al_data_ei._info["theta"]
            theta_hyb = al_data_hyb._info["theta"]
    
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            cp = ax[0].contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap="RdGy")
            ax[0].scatter(
                theta_ei[:, 0], theta_ei[:, 1], c="black", marker="+", s=ms, zorder=2
            )
            ax[0].scatter(
                thetainit[:, 0],
                thetainit[:, 1],
                zorder=2,
                marker="o",
                facecolors="none",
                edgecolors="blue",
            )
            ax[0].set_xlabel(r"$\theta_1$", fontsize=ft)
            ax[0].set_ylabel(r"$\theta_2$", fontsize=ft)
            ax[0].tick_params(axis="both", labelsize=ft)
    
            cp = ax[1].contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap="RdGy")
            ax[1].scatter(
                theta_hyb[:, 0], theta_hyb[:, 1], c="black", marker="+", s=ms, zorder=2
            )
            ax[1].scatter(
                thetainit[:, 0],
                thetainit[:, 1],
                zorder=2,
                marker="o",
                facecolors="none",
                edgecolors="blue",
            )
            ax[1].set_xlabel(r"$\theta_1$", fontsize=ft)
            ax[1].set_ylabel(r"$\theta_2$", fontsize=ft)
            ax[1].tick_params(axis="both", labelsize=ft)
            plt.savefig("Figure2.jpg", format="jpeg", bbox_inches="tight", dpi=500)
            plt.show()
    
    end = time.time()
    print("Elapsed time =", round(end - start, 3))
