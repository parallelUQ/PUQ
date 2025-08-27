"""Contains acquisition functions."""

import numpy as np
from PUQ.designmethods.gen_funcs.batch_acquisition_funcs_support import (
    build_emulator,
    compute_ivar,
    impute,
    impute_CL,
    multiple_determinants,
    multiple_pdfs,
)
from PUQ.surrogatemethods.pcHetGP import update


def get_pred(cL, emu, x, ttest, reps):

    cP = emu.predict(x=x, theta=cL)
    var_cand = cP._info["var_o"] + cP._info["nugs_o"] / reps

    testP = emu.predict(x=x, theta=ttest, thetaprime=cL)
    mu, S, cov = testP._info["mean"], testP._info["S"], testP._info["cov_o"]

    mut = mu.T
    St = np.transpose(S, (2, 0, 1))

    return mut, St, cov, var_cand


def last_iteration(x, obs, obsvar, tE, fE, des_obj):
    tmesh, skip, pc_settings = des_obj.tmesh, des_obj.skip, des_obj.pc_settings
    if skip:
        ivar = 0
    else:
        emu = build_emulator(x=x, theta=tE, f=fE, pcset=pc_settings)
        ivar = compute_ivar(emu=emu, ttest=tmesh, x=x, obs=obs, obsvar=obsvar)
    return ivar


def impute_strategy(ct, x, fE, tE, emu, liar, des_obj):

    rep = des_obj.rep
    impute_str = des_obj.impute_str
    rand_stream = des_obj.rand_stream
    pc_settings = des_obj.pc_settings

    if impute_str == "update":
        update(emu._info, x=x, X0new=ct, mult=rep)
        fE, tE = impute(
            ct=ct, x=x, fE=fE, tE=tE, reps=rep, emu=emu, rnd_str=rand_stream
        )
    else:
        if impute_str == "KB":
            fE, tE = impute(
                ct=ct, x=x, fE=fE, tE=tE, reps=rep, emu=emu, rnd_str=rand_stream
            )
        elif impute_str == "CL":
            fE, tE = impute_CL(ct=ct, x=x, fE=fE, tE=tE, reps=rep, liar=liar)

        emu = build_emulator(x=x, theta=tE, f=fE, pcset=pc_settings)

    return fE, tE, emu


class acquire:
    def __init__(
        self,
        bnew,
        rep,
        emu,
        func_cls,
        theta_mesh,
        prior,
        method="simse",
        nL=200,
        pc_settings={},
        rand_stream=None,
        impute_str="KB",
        skip=False,
    ):

        self.bnew = bnew
        self.rep = rep
        self.emu = emu
        self.x = func_cls.x
        self.tmesh = theta_mesh
        self.obs = func_cls.real_data
        self.obsvar = func_cls.obsvar
        self.method = method
        self.nL = nL
        self.prior = prior
        self.pc_settings = pc_settings
        self.func_cls = func_cls
        self.rand_stream = rand_stream
        self.impute_str = "KB" if impute_str is None else impute_str
        self.skip = False if skip is None else skip

        # print("Explore rule: ", self.method, " with ", self.impute_str)

    def acquire_new(self):
        if self.method == "simse":
            self.simse()
        elif self.method == "seivar":
            self.seivar()
        elif self.method == "var":
            self.var()

        return

    def seivar(self):

        x = self.x
        obs = self.obs
        obsvar = self.obsvar
        n_x = x.shape[0]
        # 1 x d x d
        obsvar3d = obsvar.reshape(1, n_x, n_x)

        nm = self.tmesh.shape[0]
        p = self.tmesh.shape[1]

        # Create a candidate list
        nL = self.nL
        cL = self.prior.rnd(self.nL, self.rand_stream)

        # Get emulator
        emu = self.emu
        # cov  : q x nm x nL
        # cvar : q x nL
        # mu   : nm x d
        # S    : nm x d x d
        mu, S, cov, cvar = get_pred(
            cL=cL, emu=emu, x=x, ttest=self.tmesh, reps=self.rep
        )
        V1 = S + obsvar3d

        fE = emu._info["f"]
        tE = emu._info["theta"]

        liar = np.mean(fE, axis=1)

        tnew = []
        for i in range(self.bnew):
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
                part2 = rpdf / (coef * np.sqrt(dets))

                cval = -np.sum(part2)
                vals.append(cval)

            idc = np.argmin(vals)
            ct = cL[idc, :].reshape((1, p))
            tnew.append(ct)

            fE, tE, emu = impute_strategy(ct, x, fE, tE, emu, liar, self)

            cL = self.prior.rnd(nL, self.rand_stream)

            mu, S, cov, cvar = get_pred(
                cL=cL, emu=emu, x=x, ttest=self.tmesh, reps=self.rep
            )
            V1 = S + obsvar3d

        tnew = np.array(tnew).reshape((self.bnew, p))
        ivar = last_iteration(x, obs, obsvar, tE, fE, self)
        self.tnew = tnew
        self.ivar = ivar
        return

    def simse(self):

        x = self.x
        obs = self.obs
        obsvar = self.obsvar
        n_x = x.shape[0]
        obsvar3d = obsvar.reshape(1, n_x, n_x)

        nm = self.tmesh.shape[0]
        p = self.tmesh.shape[1]

        # Create a candidate list
        nL = self.nL
        cL = self.prior.rnd(self.nL, self.rand_stream)

        # Get emulator
        emu = self.emu

        # cov  : q x nm x nL
        # cvar : q x nL
        _, _, cov, cvar = get_pred(cL=cL, emu=emu, x=x, ttest=self.tmesh, reps=self.rep)
        fE = emu._info["f"]
        tE = emu._info["theta"]

        liar = np.mean(fE, axis=1)

        tnew = []
        for i in range(self.bnew):
            G = emu._info["G"]
            B = emu._info["B"]
            GB = G @ B
            BTG = B.T @ G

            d = G.shape[0]
            q = B.shape[1]

            tau = np.zeros((nm, q, q, nL))
            phi = np.zeros((nm, d, d, nL))

            for j in range(0, q):
                tau[:, j, j, :] = cov[j, :, :] ** 2 / cvar[j, :]

            for k in range(0, nL):
                phi[:, :, :, k] = GB @ tau[:, :, :, k] @ BTG

            phidiag = np.diagonal(phi, axis1=1, axis2=2)

            vals = []
            for k in range(0, nL):
                cval = -np.sum(phidiag[:, k, :])
                vals.append(cval)

            idc = np.argmin(vals)
            ct = cL[idc, :].reshape((1, p))
            tnew.append(ct)

            fE, tE, emu = impute_strategy(ct, x, fE, tE, emu, liar, self)

            cL = self.prior.rnd(nL, self.rand_stream)
            _, _, cov, cvar = get_pred(
                cL=cL, emu=emu, x=x, ttest=self.tmesh, reps=self.rep
            )
        tnew = np.array(tnew).reshape((self.bnew, p))
        ivar = last_iteration(x, obs, obsvar, tE, fE, self)
        self.tnew = tnew
        self.ivar = ivar
        return

    def var(self):
        x = self.x
        obs = self.obs
        obsvar = self.obsvar
        n_x = x.shape[0]
        # 1 x d x d
        obsvar3d = obsvar.reshape(1, n_x, n_x)

        p = self.tmesh.shape[1]

        # Create a candidate list
        nL = self.nL
        cL = self.prior.rnd(self.nL, self.rand_stream)

        # Get emulator
        emu = self.emu

        fE = emu._info["f"]
        tE = emu._info["theta"]

        liar = np.mean(fE, axis=1)

        tnew = []
        det = np.linalg.det(obsvar)

        for i in range(self.bnew):
            pcL = emu.predict(x=x, theta=cL)
            mu, S = pcL._info["mean"].T, pcL._info["S"]
            St = np.transpose(S, (2, 0, 1))

            M = St + 0.5 * obsvar3d
            N = St + obsvar3d
            f = multiple_pdfs(obs, mu, M)
            g = multiple_pdfs(obs, mu, N)

            coef = 1 / ((2**n_x) * (np.sqrt(np.pi) ** n_x) * np.sqrt(det))

            vals = coef * f - g**2

            idc = np.argmax(vals)
            ct = cL[idc, :].reshape((1, p))
            tnew.append(ct)

            fE, tE, emu = impute_strategy(ct, x, fE, tE, emu, liar, self)

            cL = self.prior.rnd(nL, self.rand_stream)

        tnew = np.array(tnew).reshape((self.bnew, p))
        ivar = last_iteration(x, obs, obsvar, tE, fE, self)
        self.tnew = tnew
        self.ivar = ivar
        return
