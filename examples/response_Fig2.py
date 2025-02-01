import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen
from PUQ.design import designer
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from PUQ.surrogatemethods.covariances import cov_gen
from scipy.linalg import cholesky
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    multiple_pdfs,
    build_emulator,
)
import scipy.stats as sps

def update_alloc(fitinfo, x, X0new=None, mult=None, munew=None):
    numGPs = fitinfo["numGPs"]

    Gi = fitinfo["Gi"]
    h = fitinfo["h"]
    mu_ct = 1 * munew

    mu_ct = Gi @ (mu_ct - h)
    Z0new = mu_ct.T

    for i in range(0, numGPs):
        info = fitinfo["emulist"][i]
        for cid, ct in enumerate(X0new):
            if mult[cid] > 0:
                ids = np.all(info["X0"] == ct, axis=1)
                info["Z0"][ids] = (
                    info["Z0"][ids] * info["mult"][ids] + Z0new[0, i] * mult[cid]
                ) / (info["mult"][ids] + mult[cid])
                info["mult"][ids] += mult[cid]

        if info["is_homGP"] == True:
            C = cov_gen(X1=info["X0"], theta=info["theta"])
            Ki = cholesky(C + np.diag(info["eps"] + info["g"] / info["mult"]))
            Ki = np.linalg.inv((Ki))
            info["Ki"] = Ki @ Ki.T
        else:
            Cg = cov_gen(X1=info["X0"], theta=info["theta_g"])
            Kg_c = cholesky(Cg + np.diag(info["eps"] + info["g"] / info["mult"]))
            Kgi = np.linalg.inv((Kg_c))
            Kgi = Kgi @ Kgi.T
            M = np.dot(Cg, np.dot(Kgi, info["Delta"] - info["nmean"]))
            Lambda = info["nmean"] + M

            if info["logN"]:
                Lambda = np.exp(Lambda)
            else:
                Lambda[Lambda <= 0] = info["eps"]

            info["Lambda"] = Lambda

            C = cov_gen(X1=info["X0"], theta=info["theta"])
            Ki = cholesky(C + np.diag(info["Lambda"] / info["mult"] + info["eps"]))
            Ki = np.linalg.inv((Ki))
            info["Ki"] = Ki @ Ki.T

ft = 18

def observe_posterior(
    Xpl,
    Ypl,
    emu=None,
    theta_test=None,
    cls_func=None,
    tnew=None,
    repnew=None,
    labelfig=None,
    predold=None,
    phat=None,
):

    if phat is None:
        testP = emu.predict(x=x, theta=theta_test)

        mu, S = testP._info["mean"], testP._info["S"]

        if predold is not None:
            mu = predold._info["mean"]

        mut = mu.T
        St = np.transpose(S, (2, 0, 1))
        n_x = x.shape[0]

        # 1 x d x d
        obsvar3d = cls_func.obsvar.reshape(1, n_x, n_x)
        V1 = St + obsvar3d
        phat = np.zeros(len(theta_test))
        phat = multiple_pdfs(cls_func.real_data, mut, V1)

        # for tid in range(0, len(theta_test)):
        #     rnd = sps.multivariate_normal(mean=mut[tid, :], cov=V1[tid, :, :])
        #     phat[tid] = rnd.pdf(cls_func.real_data)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(Xpl, Ypl, phat.reshape(50, 50), 20, cmap="coolwarm")

    if tnew is not None:
        for label, x_count, y_count in zip(repnew, tnew[:, 0], tnew[:, 1]):
            if label <= 2:
                cs = "cyan"
            else:
                cs = "black"
            plt.annotate(
                label,
                xy=(x_count, y_count),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=16,
                color=cs,
            )

    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$\theta_1$", fontsize=ft)
    ax.set_ylabel(r"$\theta_2$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    plt.savefig(labelfig, bbox_inches="tight")
    plt.show()


def emu_observe(Xpl, Ypl, meanmesh):

    for i in range(0, 2):
        fig, ax = plt.subplots(figsize=(5, 5))
        cs = ax.contour(Xpl, Ypl, meanmesh[i, :].reshape(50, 50), 20, cmap="coolwarm")
        cbar = fig.colorbar(cs)
        ax.set_xticks([0, 0.5, 1])  # Custom tick locations for x-axis
        ax.set_yticks([0, 0.5, 1])  # Custom tick locations for y-axis
        ax.set_xlabel(r"$\theta_1$", fontsize=ft)
        ax.set_ylabel(r"$\theta_2$", fontsize=ft)
        ax.tick_params(axis="both", labelsize=ft)
        plt.show()


args = parse_arguments()

emu_visual = False
# # # # #
batch = 16
maxiter = 1 * batch
workers = batch + 1
seeds = [0, 1]
n0 = 15
rep0 = 2
nmesh = 50
rho = 1
    
if __name__ == "__main__":

    for funcid, func in enumerate(["banana", "bimodal"]):
        for s in np.arange(seeds[funcid], seeds[funcid] + 1):

            cls_func = eval(func)()
            cls_func.realdata(seed=None)

            theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
            test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

            observe_posterior(
                Xpl=Xpl,
                Ypl=Ypl,
                cls_func=cls_func,
                labelfig="true" + func + ".png",
                phat=p_test,
            )

            if emu_visual:
                emu_observe(Xpl, Ypl, f_test.T)

            # Set a uniform prior
            prior_func = prior_dist(dist="uniform")(
                a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
            )

            # Set random stream for initial design
            persis_info = {"rand_stream": np.random.default_rng(s)}

            # Initial sample
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
            theta0 = sampling(n0)
            theta0 = np.repeat(theta0, rep0, axis=0)
            f0 = np.zeros((cls_func.d, n0 * rep0))
            for i in range(0, n0 * rep0):
                f0[:, i] = cls_func.sim_f(theta0[i, :], persis_info=persis_info)

            x = cls_func.x

            emu = build_emulator(
                x=x, theta=theta0, f=f0, pcset={"standardize": True, "latent": False}
            )

            al_ivar = designer(
                data_cls=cls_func,
                method="p_sto_bseq",
                acquisition="seivar",
                args={
                    "prior": prior_func,
                    "data_test": test_data,
                    "max_iter": maxiter,
                    "nworkers": workers,
                    "batch_size": batch,
                    "des_init": {"seed": s, "theta": theta0, "f": f0},
                    "alloc_settings": {
                        "method": "ivar",
                        "use_Ki": True,
                        "rho": rho,
                        "theta": None,
                        "a0": None,
                        "gen": False,
                    },
                    "pc_settings": {"standardize": True, "latent": False},
                    "des_settings": {
                        "is_exploit": True,
                        "is_explore": False,
                        "nL": 200,
                        "impute_str": "update",
                    },
                },
            )

            predmesh = emu.predict(x=x, theta=theta_test)
            meanmesh = predmesh.mean()

            if emu_visual:
                emu_observe(Xpl, Ypl, meanmesh)

            theteanew = al_ivar._info["theta"][n0 * rep0 :, :]
            unique_theta, counts = np.unique(theteanew, axis=0, return_counts=True)

            print("unique:", unique_theta.shape)
            print("counts:", counts)
            print(unique_theta)

            # Observe posterior
            observe_posterior(
                Xpl=Xpl,
                Ypl=Ypl,
                emu=emu,
                theta_test=theta_test,
                cls_func=cls_func,
                tnew=al_ivar._info["theta0"],
                repnew=al_ivar._info["reps0"],
                labelfig="estimated_" + func + ".png",
                predold=None,
            )

            prednew = emu.predict(x=x, theta=unique_theta)
            munew = prednew.mean()

            print("mean shape:", munew.shape)

            update_alloc(
                fitinfo=emu._info, x=x, X0new=unique_theta, mult=counts, munew=munew
            )

            # Observe posterior after update
            observe_posterior(
                Xpl=Xpl,
                Ypl=Ypl,
                emu=emu,
                theta_test=theta_test,
                cls_func=cls_func,
                tnew=None,
                repnew=None,
                labelfig="estimated_" + func + str(batch) + ".png",
                predold=None,
            )

            if emu_visual:
                predmesh = emu.predict(x=x, theta=theta_test)
                meanmesh = predmesh.mean()
                emu_observe(Xpl, Ypl, meanmesh)
