import pandas as pd
import numpy as np
import scipy.stats as sps
from PUQ.designmethods.sequential_deterministic import sequential_design
from PUQ.designmethods.utils import parse_arguments
from PUQ.prior import prior_dist
from plots import plotline
from test_funcs_new import unimodal, banana, bimodal, unidentifiable
import time
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt

if __name__ == "__main__":
    design_start = time.time()
    args = parse_arguments()
    print("Running function: " + args.funcname)
    args.funcname = "bimodal"
    # Choose the test function
    if args.funcname == "unimodal":
        cls_func = unimodal()
    elif args.funcname == "banana":
        cls_func = banana()
    elif args.funcname == "bimodal":
        cls_func = bimodal()
    elif args.funcname == "unidentifiable":
        cls_func = unidentifiable()

    # Create a mesh for test set
    xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], 50)
    ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], 50)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    thetatest = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
    ftest = np.zeros((thetatest.shape[0], cls_func.d))
    for i in range(thetatest.shape[0]):
        ftest[i, :] = cls_func.function(thetatest[i,0], thetatest[i,1])
            
    ptest = np.zeros(thetatest.shape[0])
    if cls_func.data_name == "unimodal":
        ptest = sps.norm.pdf(cls_func.real_data - ftest, 0, np.sqrt(cls_func.obsvar))
    else:
        for i in range(ftest.shape[0]):
            mean = ftest[i, :]
            rnd = sps.multivariate_normal(mean=mean, cov=cls_func.obsvar)
            ptest[i] = rnd.pdf(cls_func.real_data)
    test_data = {"theta": thetatest, "f": ftest, "p": ptest, "p_prior": 1}
    
    # Set a uniform prior
    prior_func = prior_dist(dist="uniform")(
        a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
    )

    # Define acquisition functions
    # acq_funcs = ["eivar", "rnd", "maxvar", "maxexp"]
    acq_funcs = ["ivar", "rnd", "var"]
    datalist = []
    rep_no = 2
    n0 = 10
    # Run over 50 replications
    for seed_id in range(1, rep_no + 1):
        # Initial sample
        n0 = 10
        sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(seed_id))
        t0 = sampling(n0)
        f0 = np.zeros((t0.shape[0], cls_func.d))
        for i in range(t0.shape[0]):
            f0[i, :] = cls_func.function(t0[i,0], t0[i,1])
            
        for func in acq_funcs:
            print("Running " + func + " with seed " + str(seed_id))
            des_obj = sequential_design(cls_func)
            des_obj.build_design(z0=t0, 
                                 f0=f0, 
                                 T=75, 
                                 test=test_data, 
                                 af=func,
                                 args={
                                        "mini_batch": 1,
                                        "n_init_thetas": 10,
                                        "nworkers": 2,
                                        "prior": prior_func,
                                        "data_test": test_data,
                                        "seed": seed_id,
                                        "integral": "LHS"
                                    })
        
        
            theta_al = des_obj.zs
        
            fig, ax = plt.subplots()
            cp = ax.contour(Xpl, Ypl, ptest.reshape(50, 50), 20, cmap="RdGy")
            ax.scatter(theta_al[n0:, 0], theta_al[n0:, 1], c="black", marker="+", zorder=2)
            ax.scatter(
                theta_al[0:n0, 0],
                theta_al[0:n0, 1],
                zorder=2,
                marker="o",
                facecolors="none",
                edgecolors="blue",
            )
            ax.set_xlabel(r"$\theta_1$", fontsize=16)
            ax.set_ylabel(r"$\theta_2$", fontsize=16)
            ax.tick_params(axis="both", labelsize=16)
            plt.show()

            # TV = al_unimodal._info["TV"]

            for ms_id, ms in enumerate(des_obj.H):
                d = {
                    "TV": ms["MAD"],
                    "TV_id": ms_id,
                    "rep": seed_id,
                    "batch": 1,
                    "methods": func,
                    "worker": 2,
                    "synth": args.funcname,
                    "theta_id": ms_id,
                }

                datalist.append(d)

            datalist.append(d)
    df = pd.DataFrame(datalist)


    # Create the plot and save it
    plotline(
        df,
        acq_funcs,
        rep_no,
        w=2,
        b=1,
        s=args.funcname,
        ylim=[0.000005, 0.00001],
        idstart=0,
    )
