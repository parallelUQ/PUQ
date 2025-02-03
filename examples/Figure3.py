import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments
from test_funcs import sinf
from utilities import test_data_gen_1d
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    build_emulator,
    multiple_determinants,
    multiple_pdfs,
)
from PUQ.designmethods.gen_funcs.acquire_new import get_pred

args = parse_arguments()

# # # # #
args.funcname = "sinf"
args.seedmin = 3  # 13
args.seedmax = 4  # 14
# # # # #

n0 = 6
rep0 = 10
nmesh = 50
batch = 15
reps = 5
bnew = int(batch / reps)

if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):
        cls_func = eval(args.funcname)()
        cls_func.realdata(seed=s)

        theta_test, p_test, f_test = test_data_gen_1d(cls_func, 100)
        test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

        # Set a uniform prior
        prior_func = prior_dist(dist="uniform")(
            a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
        )
        persis_info = {"rand_stream": np.random.default_rng(12345)}

        # Initial sample
        theta0u = np.linspace(
            cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], n0
        )[:, None]
        f0mean = np.zeros(len(theta0u))
        p0 = np.zeros(len(theta0u))
        theta0 = np.repeat(theta0u, rep0, axis=0)
        f0 = np.zeros(len(theta0))
        for tid, t in enumerate(theta0):
            f0[tid] = cls_func.sim_f(t, persis_info=persis_info)[0]

        for tid, t in enumerate(theta0u):
            f0mean[tid] = cls_func.function(t)[0]
            rnd = sps.norm(loc=f0mean[tid], scale=np.sqrt(cls_func.obsvar))
            p0[tid] = rnd.pdf(cls_func.real_data)[0, 0]

        n_x, p, x = cls_func.x.shape[0], theta_test.shape[1], cls_func.x
        obs, obsvar = cls_func.real_data, cls_func.obsvar
        theta_acq, n_acq = None, None
        obsvar3d = cls_func.obsvar.reshape(1, n_x, n_x)
        is_cov = False

        # Create a candidate list
        nL = 500
        nm = theta_test.shape[0]
        pc_settings = {"standardize": True, "latent": False}
        # plt.scatter(theta0, f0)
        # plt.show()

        fig, ax = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)

        emu = build_emulator(x=x, theta=theta0, f=f0[None, :], pcset=pc_settings)
        for i in range(bnew):
            ft = 18
            cL = np.linspace(
                cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nL
            )[:, None]

            mu, S, cov, cvar = get_pred(
                cL=cL, emu=emu, x=x, ttest=theta_test, reps=reps
            )
            V1 = S + obsvar3d
            V2 = S + 0.5 * obsvar3d

            G = emu._info["G"]
            B = emu._info["B"]
            GB = G @ B
            BTG = B.T @ G

            d = G.shape[0]
            q = B.shape[1]

            tau = np.zeros((nm, q, q, nL))
            phi = np.zeros((nm, d, d, nL))

            coef = 1 / ((2**d) * (np.sqrt(np.pi) ** d))

            for j in range(0, q):
                tau[:, j, j, :] = cov[j, :, :] ** 2 / cvar[j, :]

            for k in range(0, nL):
                phi[:, :, :, k] = GB @ tau[:, :, :, k] @ BTG

            vals = []
            for k in range(0, nL):
                # C1: nm x d x d
                # C2: nm x d x d
                phic = phi[:, x, x.T, k]
                C1 = (V1 + phic) * 0.5
                C2 = V1 - phic

                rpdf = multiple_pdfs(obs, mu, C1)
                dets = multiple_determinants(C2)
                part2 = rpdf / np.sqrt(dets)

                rpdf2 = multiple_pdfs(obs, mu, V2)
                denum2 = obsvar

                cval = coef * (np.sum(rpdf2 / np.sqrt(denum2)) - np.sum(part2))

                vals.append(cval)

            idc = np.argmin(vals)
            minacq = np.min(vals)
            cu = cL[idc, :].reshape((1, p))

            ax[0, i].plot(cL, vals, color="blue")
            ax[0, i].set_xlabel(r"$\theta$", fontsize=ft)
            ax[0, i].set_ylabel("IVAR", fontsize=ft)
            ax[0, i].tick_params(axis="both", labelsize=ft)
            ax[0, i].scatter(cu, minacq, color="green", marker="*", s=200)

            ctheta = np.repeat(cu, reps, axis=0)
            cpred = emu.predict(x=x, theta=ctheta)

            cmean = cpred.mean()
            cnoise = cpred._info["var_noisy"]
            fnoise = persis_info["rand_stream"].normal(
                loc=cmean[0], scale=np.sqrt(cnoise[0]), size=reps
            )

            theta0 = np.concatenate([theta0, ctheta], axis=0)
            f0 = np.concatenate([f0, fnoise.flatten()])

            fc = np.zeros(len(cu))
            pc = np.zeros(len(cu))
            for tid, t in enumerate(cu):
                fc[tid] = cls_func.function(t)
                rnd = sps.norm(loc=fc[tid], scale=np.sqrt(cls_func.obsvar))
                pc[tid] = rnd.pdf(cls_func.real_data)

            phat = np.zeros(theta_test.shape[0])
            phatvar = np.zeros(theta_test.shape[0])
            pvar1 = np.zeros(theta_test.shape[0])
            for tid in range(0, len(theta_test)):
                rnd = sps.norm(loc=mu[tid, 0], scale=np.sqrt(obsvar + S[tid, 0, 0]))
                phat[tid] = rnd.pdf(obs)
                rnd = sps.norm(
                    loc=mu[tid, 0], scale=np.sqrt(0.5 * obsvar + S[tid, 0, 0])
                )
                pvar1[tid] = rnd.pdf(obs)
                phatvar[tid] = (1 / (2 * np.sqrt(np.pi) * np.sqrt(obsvar))) * pvar1[
                    tid
                ] - phat[tid] ** 2

            ax[1, i].plot(theta_test, p_test, color="red")
            ax[1, i].plot(
                theta_test, phat, color="blue", linestyle="dashed", linewidth=2.5
            )
            ax[1, i].fill_between(
                theta_test.flatten(),
                (phat - np.sqrt(phatvar)).flatten(),
                (phat + np.sqrt(phatvar)).flatten(),
                color="blue",
                alpha=0.1,
            )
            ax[1, i].set_xlabel(r"$\theta$", fontsize=ft)
            ax[1, i].set_ylabel(r"$p(y|\theta)$", fontsize=ft)
            ax[1, i].tick_params(axis="both", labelsize=ft)
            ax[1, i].scatter(theta0u, p0, color="black", s=100)
            ax[1, i].scatter(cu, pc, color="green", marker="*", s=200)

            theta0u = np.concatenate([theta0u, cu], axis=0)
            p0 = np.concatenate([p0, pc.flatten()])

            print("Adding point:", np.round(ctheta, 2))

            # emu = build_emulator(x=x, theta=theta0, f=f0[None, :], pcset=pc_settings)
            from PUQ.surrogatemethods.pcHetGP import update

            update(emu._info, x=x, X0new=cu, mult=reps)

        plt.savefig("Figure3.png", bbox_inches="tight")
        plt.show()
