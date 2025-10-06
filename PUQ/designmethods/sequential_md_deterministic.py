import numpy as np
from PUQ.designmethods.support import multiple_pdfs, multiple_determinants
from PUQ.designmethods.gen_funcs.acquisition_md_deterministic import (
    ivar,
    var,
    imse,
    rnd,
    exp,
)
import time
from PUQ.surrogate import emulator


class sequential_design:
    def __init__(self, cls_func, trace=False):
        self.cls_func = cls_func
        self.trace = trace
        self.y = self.cls_func.real_data
        self.d = self.cls_func.d
        self.Sigma = self.cls_func.obsvar
        self.Sigma3d = self.Sigma.reshape(1, self.d, self.d)
        self.detSigma = np.linalg.det(self.Sigma)
        return

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def get(self, key):
        return self.__dict__.get(key)

    def build_design(self, z0, f0, T, af, test=None, args={}):

        self.test = test
        args["rand_stream"] = np.random.default_rng(args["seed"])
        H = []
        timel = []
        metric_sum = {}

        md = np.arange(f0.shape[1]).reshape(f0.shape[1], 1)
        for t in range(0, T):
            print(f"t: {t}") if self.trace else None

            tic = time.time()

            model = emulator(
                x=z0,
                theta=md,
                f=f0,
                method="multihomGP",
                args={
                    "lower": None,
                    "upper": None,
                    "noiseControl": {
                        "k_theta_g_bounds": (1, 100),
                        "g_max": 1e2,
                        "g_bounds": (1e-6, 1),
                    },
                    "init": {},
                    "known": {},
                    "settings": {
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
                },
            )

            toc = time.time()
            timel.append(toc - tic)

            args["seed"] += 1

            if test is not None:
                metric_sum = self.eval_perf(model)

            acq_func = eval(af)(model=model, cls_func=self.cls_func, args=args)

            acq_func.acquire_new()
            tnew = acq_func.znew

            fnew = self.cls_func.function(tnew.flatten()[0], tnew.flatten()[1]).reshape(
                1, self.cls_func.d
            )

            f0 = np.concatenate((f0, fnew), axis=0)
            z0 = np.concatenate((z0, tnew), axis=0)

            H.append({"t": t, "f": fnew, "z": tnew, "MAD": metric_sum.get("MAD", None)})

        unique_rows, counts = np.unique(z0, axis=0, return_counts=True)

        self.t = unique_rows
        self.reps = counts
        self.fs = f0
        self.zs = z0
        self.time = timel
        self.H = H

        return self

    def eval_perf(self, model):

        nm, d = self.test["theta"].shape[0], self.cls_func.d

        # predict at mesh
        pr = model.predict(x=self.test["theta"], thetaprime=self.test["theta"])

        # ntot, ntot x ntot, ntot
        mu, Sn = pr._info["mean"], pr._info["S"]

        muT = mu.reshape(nm, d)
        S = Sn.transpose(2, 0, 1)  # Sn[self.idr, self.idc].reshape(nm, d, d)

        N = S + self.Sigma3d
        g = multiple_pdfs(self.y, muT, N)

        MSE = np.mean(((g.flatten() - self.test["p"].flatten()) ** 2))
        MAD = np.mean(np.abs(g.flatten() - self.test["p"].flatten()))

        metric_sum = {"MAD": MAD, "MSE": MSE}
        return metric_sum
