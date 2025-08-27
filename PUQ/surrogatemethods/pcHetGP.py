from PUQ.surrogate import emulator
from PUQ.surrogatemethods.hetGP import predict as predict_hetGP
import numpy as np
from PUQ.surrogatemethods.covariances import cov_gen
from scipy.linalg import cholesky, inv


def fit(
    fitinfo,
    x,
    theta,
    f,
    lower=None,
    upper=None,
    noiseControl={"k_theta_g_bounds": (1, 100), "g_max": 1e2, "g_bounds": (1e-6, 1)},
    init={},
    known={},
    settings={
        "linkThetas": "joint",
        "logN": True,
        "initStrategy": "residuals",
        "checkHom": True,
        "penalty": True,
        "trace": 0,
        "return.matrices": True,
        "return.hom": False,
        "factr": 1e9,
    },
    covtype="Gaussian",
    pc_settings={"standardize": True, "latent": False},
    **kwargs
):

    numGPs = f.shape[0]
    h = np.zeros((numGPs, 1))
    s = np.ones((numGPs))
    fs = np.zeros(f.shape)

    if pc_settings["standardize"]:
        # print("Standardizing")
        fitinfo["standardize"] = True
        for k in range(0, numGPs):
            h[k, 0] = np.mean(f[k, :])
            s[k] = np.std(f[k, :])

    G = np.diag(s)
    Gi = np.diag(1 / s)
    fs = Gi @ (f - h)
    B = np.eye(numGPs)

    if pc_settings["latent"]:
        # print("Dimension reduction")
        fitinfo["latent"] = True

        U, S, _ = np.linalg.svd(fs, full_matrices=False)
        exp_var = np.cumsum(S**2) / np.cumsum(S**2)[-1]
        numGPs = int(np.argwhere(np.array(exp_var) > 0.99)[0]) + 1

        # d x q orthogonal matrix
        B = U[:, 0:numGPs]

        # trasform q x n
        W = B.T @ fs

    emulist = [dict() for x in range(0, numGPs)]
    for i in range(0, numGPs):
        if pc_settings["latent"]:
            print("Latent surrogate")
            fi = W[i, :][None, :]
        else:
            # print("On site surrogate")
            fi = fs[i, :][None, :]
        emu = emulator(
            x=np.array([[i]]),
            theta=theta,
            f=fi,
            method="hetGP",
            args={
                "noiseControl": noiseControl,
                "lower": lower,
                "upper": upper,
                "settings": settings,
                "init": init,
                "known": known,
                "covtype": covtype,
            },
        )

        emulist[i] = emu._info

    fitinfo["theta"] = theta
    fitinfo["f"] = f
    fitinfo["emulist"] = emulist
    fitinfo["numGPs"] = numGPs
    fitinfo["fs"] = fs
    fitinfo["h"] = h
    fitinfo["G"] = G
    fitinfo["Gi"] = Gi
    fitinfo["B"] = B

    return


def predict(predinfo, fitinfo, x, theta, thetaprime, **kwargs):

    numGPs = fitinfo["numGPs"]
    d = fitinfo["fs"].shape[0]
    n = theta.shape[0]

    # calculate predictive mean and variance
    mean_pc = np.full((numGPs, n), np.nan)
    var_pc = np.full((numGPs, n), np.nan)
    nugs_pc = np.full((numGPs, n), np.nan)

    if thetaprime is not None:
        npr = thetaprime.shape[0]
        covmat_pc = np.full((numGPs, n, npr), np.nan)
    for i in range(0, numGPs):
        predinfo_hetGP = {}
        info = fitinfo["emulist"][i]
        predict_hetGP(
            predinfo=predinfo_hetGP,
            fitinfo=info,
            x=np.array([[i]]),
            theta=theta,
            thetaprime=thetaprime,
        )

        mean_pc[i, :] = predinfo_hetGP["mean"]
        var_pc[i, :] = predinfo_hetGP["var"]
        nugs_pc[i, :] = predinfo_hetGP["nugs"]

        if thetaprime is not None:
            covmat_pc[i, :, :] = predinfo_hetGP["covmat"]

    # calculate predictive mean and variance
    predinfo["mean_o"] = 1 * mean_pc
    predinfo["var_o"] = 1 * var_pc
    predinfo["nugs_o"] = 1 * nugs_pc
    if thetaprime is not None:
        predinfo["cov_o"] = 1 * covmat_pc

    predinfo["S"] = np.full((d, d, n), np.nan)
    predinfo["R"] = np.full((d, d, n), np.nan)
    predinfo["var"] = np.full((d, n), np.nan)
    predinfo["nugs"] = np.full((d, n), np.nan)
    predinfo["var_noisy"] = np.full((d, n), np.nan)
    if thetaprime is not None:
        predinfo["covmat"] = np.full((d, n, npr), np.nan)
    predinfo["mean"] = fitinfo["h"] + fitinfo["G"] @ (fitinfo["B"] @ mean_pc)

    for i in range(0, theta.shape[0]):
        C = np.diag(var_pc[:, i])
        R = np.diag(nugs_pc[:, i])
        predinfo["S"][:, :, i] = (
            fitinfo["G"] @ fitinfo["B"] @ C @ fitinfo["B"].T @ fitinfo["G"]
        )
        predinfo["R"][:, :, i] = (
            fitinfo["G"] @ fitinfo["B"] @ R @ fitinfo["B"].T @ fitinfo["G"]
        )
        predinfo["var"][:, i] = np.diag(predinfo["S"][:, :, i])
        predinfo["nugs"][:, i] = np.diag(predinfo["R"][:, :, i])

    predinfo["var_noisy"] = predinfo["var"] + predinfo["nugs"]

    return predinfo


def update(fitinfo, x, X0new=None, mult=None):
    numGPs = fitinfo["numGPs"]

    Gi = fitinfo["Gi"]
    h = fitinfo["h"]
    pred_ct = predict(predinfo={}, fitinfo=fitinfo, x=x, theta=X0new, thetaprime=None)
    mu_ct = pred_ct["mean"]
    mu_ct = Gi @ (mu_ct - h)
    Z0new = mu_ct.T
    # print(mu_ct.shape)

    for i in range(0, numGPs):
        info = fitinfo["emulist"][i]

        if info["is_homGP"] == True:
            info["X0"] = np.concatenate((info["X0"], X0new), axis=0)
            info["Z0"] = np.concatenate((info["Z0"], np.array([Z0new[0, i]])))
            info["mult"] = np.concatenate((info["mult"], np.array([mult])))

            C = cov_gen(X1=info["X0"], theta=info["theta"])
            Ki = cholesky(C + np.diag(info["eps"] + info["g"] / info["mult"]))
            Ki = np.linalg.inv((Ki))
            info["Ki"] = Ki @ Ki.T
        else:
            Cg = cov_gen(X1=info["X0"], theta=info["theta_g"])
            Kg_c = cholesky(Cg + np.diag(info["eps"] + info["g"] / info["mult"]))
            Kgi = np.linalg.inv((Kg_c))
            Kgi = Kgi @ Kgi.T

            kg = cov_gen(X1=X0new, X2=info["X0"], theta=info["theta_g"])
            M = np.dot(kg, np.dot(Kgi, info["Delta"] - info["nmean"]))

            info["X0"] = np.concatenate((info["X0"], X0new), axis=0)
            info["Z0"] = np.concatenate((info["Z0"], np.array([Z0new[0, i]])))
            info["mult"] = np.concatenate((info["mult"], np.array([mult])))
            info["Delta"] = np.concatenate((info["Delta"], M + info["nmean"]))
            # Updated
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
