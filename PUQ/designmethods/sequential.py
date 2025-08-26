from hetgpy import hetGP
import numpy as np
import matplotlib.pyplot as plt
from PUQ.designmethods.support import multiple_pdfs, multiple_determinants
from PUQ.designmethods.gen_funcs.acquisition import var, ivar, imse, lookahead
import time


class sequential_design:
    def __init__(self, cls_func, trace=True):
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

    def build_design(self, z0, f0, T, persis_info, af, test=None, args={}):

        if test is not None:
            self.test_data_gen(test)

        H = []
        timel = []
        metric_sum = {}
        for t in range(0, T):
            # print(f"t: {t}") if self.trace else None
            print(t)
            tic = time.time()

            # hetGP
            model = hetGP()
            model.mle(X=z0, 
                      Z=f0, 
                      covtype="Gaussian", 
                      known={"beta0":np.mean(f0)})

            toc = time.time()
            timel.append(toc - tic)
            print(toc - tic)
            args["seed"] += 1
            
            if test is not None:
                metric_sum = self.eval_perf(model, args.get("extra_metric", True))

            acq_func = eval(af)(model=model, cls_func=self.cls_func, args=args)

            acq_func.acquire_new()
            znew = acq_func.znew
            fnew = self.cls_func.sim_f(znew.flatten(), persis_info=persis_info).reshape(
                1, 1
            )

            f0 = np.concatenate((f0, fnew), axis=0)
            z0 = np.concatenate((z0, znew), axis=0)

            H.append(
                {
                    "t": t,
                    "new": acq_func.new,
                    "h": getattr(acq_func, "h", None),
                    "f": fnew,
                    "z": znew,
                    "MSE": metric_sum.get("MSE", None),
                    "MAD": metric_sum.get("MAD", None),
                    "VAR": metric_sum.get("VAR", None),
                    "MSEy": metric_sum.get("MSEy", None),
                    "MADy": metric_sum.get("MADy", None),
                    "MSEn": metric_sum.get("MSEn", None),
                    "MADn": metric_sum.get("MADn", None),
                    "summary_metric": metric_sum.get("sum_metric", None),
                }
            )

        unique_rows, counts = np.unique(z0, axis=0, return_counts=True)
        
        self.xt = unique_rows
        self.reps = counts
        self.fs = f0
        self.zs = z0
        self.time = timel
        self.H = H

        return self

    def test_data_gen(self, test):

        # Gen test data
        self.ttest, self.ptest, self.wtest, self.ntest, self.ftest = (
            test["theta"],
            test["p"],
            test["w"],
            test["noise"],
            test["f"],
        )
        # plt.scatter(self.ttest[:, 0], self.ttest[:, 1], c=self.ftest)
        # plt.show()

        # plt.scatter(self.ttest[:, 0], self.ttest[:, 1], c=self.ntest)
        # plt.show()

        x_tiled = np.tile(self.cls_func.x, (self.ttest.shape[0], 1))
        t_repeated = np.repeat(self.ttest, self.cls_func.x.shape[0], axis=0)
        self.ztest = np.hstack([x_tiled, t_repeated])

        # To be used for performance evaluations
        ntot, nm, d = self.ztest.shape[0], self.ttest.shape[0], self.cls_func.d

        # to construct S matrix
        self.idr = np.arange(0, ntot)[:, None]
        idc = np.arange(0, ntot).reshape(nm, d)
        self.idc = np.repeat(idc, repeats=d, axis=0)

    def eval_perf(self, model, extra_metric=False):

        nm, d = self.ttest.shape[0], self.cls_func.d
        # sc = plt.scatter(self.ztest[:, 1], self.ztest[:, 2], c=self.wtest, cmap="viridis")  # Use any colormap
        # plt.colorbar(sc, label="wtest values")  # Add a colorbar
        # plt.show()

        # predict at mesh
        pr = model.predict(x=self.ztest, xprime=self.ztest)

        # ntot, ntot x ntot, ntot
        mu, Sn, nugs = pr["mean"], pr["cov"], pr["nugs"]
        
        muT = mu.reshape(nm, d)
        S = Sn[self.idr, self.idc].reshape(nm, d, d)

        N = S + self.Sigma3d
        g = multiple_pdfs(self.y, muT, N)

        MSE = np.mean(((g.flatten() - self.ptest.flatten()) ** 2) * self.wtest)
        MAD = np.mean(np.abs(g.flatten() - self.ptest.flatten()) * self.wtest)

        M = S + 0.5 * self.Sigma3d
        f = multiple_pdfs(self.y, muT, M)

        twopiddet = (2**self.d) * (np.sqrt(np.pi) ** self.d) * np.sqrt(self.detSigma)
        VAR = np.mean(((1 / twopiddet) * f - g**2) * self.wtest)
        
        if extra_metric:
            diff_y = (mu.flatten() - self.ftest.flatten()).reshape(nm, d)
            diff_n = (nugs.flatten() - self.ntest.flatten()).reshape(nm, d)
        
     
            MSEy = np.mean(np.mean(diff_y**2, axis=1).flatten() * self.wtest) 
            MADy = np.mean(np.mean(np.abs(diff_y), axis=1).flatten() * self.wtest) 
            
            MSEn = np.mean(np.mean(diff_n**2, axis=1).flatten() * self.wtest) 
            MADn = np.mean(np.mean(np.abs(diff_n), axis=1).flatten() * self.wtest) 
            
            # Weighted absolute differences
            f_w = np.mean(np.abs(diff_y) * self.wtest[:, None], axis=0)
            n_w = np.mean(np.abs(diff_n) * self.wtest[:, None], axis=0)
            
            # Unweighted absolute differences
            f_wh = np.mean(np.abs(diff_y), axis=0)
            n_wh = np.mean(np.abs(diff_n), axis=0)
            summary = {}
            for di in range(d):
                summary[f"f{di+1}w"] = f_w[di]
                summary[f"f{di+1}"]  = f_wh[di]
                summary[f"n{di+1}w"] = n_w[di]
                summary[f"n{di+1}"]  = n_wh[di]
        else:
            MSEy, MADy, MSEn, MADn, summary = 0, 0, 0, 0, 0

        metric_sum = {"MSE": MSE, "MAD": MAD, "VAR": VAR,
                     "MSEy": MSEy, "MADy": MADy, "MSEn": MSEn, "MADn": MADn, 
                     "summary": summary}
        return metric_sum
