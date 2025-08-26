import numpy as np
from test_functions import sinfunc
import scipy.stats as sps
from smt.sampling_methods import LHS
from PUQ.designmethods.sequential import sequential_design
from utils import toy_example, visual_xt

maxiter = 50
new = True
seedmin, seedmax = 2, 3
#seedmin, seedmax = 1, 2
dfl = []

if __name__ == "__main__":

    for s in np.arange(seedmin, seedmax):

        cls_data = sinfunc()
        dt = len(cls_data.theta_true)
        cls_data.realdata(x=np.array([0.22222222, 0.88888889])[:, None], seed=None)

        # test data
        nmesh, nt = 50, 100
        xpl = np.linspace(cls_data.zlim[0][0], cls_data.zlim[0][1], nmesh)
        ypl = np.linspace(cls_data.zlim[1][0], cls_data.zlim[1][1], nmesh)
        Xpl, Ypl = np.meshgrid(xpl, ypl)
        zg = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
        fg = np.array([cls_data.function(*xt) for xt in zg])[:, None]
        ng = np.array([cls_data.noise(*xt) for xt in zg])[:, None]

        tg = np.linspace(cls_data.zlim[1][0], cls_data.zlim[1][1], nt)[:, None]
        feval = np.array(
            [[cls_data.function(x, t) for x in cls_data.x] for t in tg]
        ).squeeze()
        pg = np.array(
            [
                sps.multivariate_normal(mean=f, cov=cls_data.obsvar).pdf(
                    cls_data.real_data
                )
                for f in feval
            ]
        )[:, None]
        
        # Set random stream for initial design
        persis_info = {"rand_stream": np.random.default_rng(s)}
        
        # Visualize
        toy_example(cls_data, Xpl, Ypl, fg, ng, persis_info)

        if new:
            # Initial sample
            n0, rep0 = 20, 5
            sampling = LHS(xlimits=cls_data.zlim, random_state=int(s))
            z0u = sampling(n0)
            z0 = np.repeat(z0u, rep0, axis=0)
            f0 = np.array([cls_data.sim_f(z, persis_info=persis_info) for z in z0])
        else:
            # Initial sample
            grid_size = 10
            rep0 = 5
            z1 = np.linspace(cls_data.zlim[1][0], cls_data.zlim[1][1], grid_size)
            z2 = np.linspace(cls_data.zlim[1][0], cls_data.zlim[1][1], grid_size)
            zm1, zm2 = np.meshgrid(z1, z2)
            z0u = np.vstack([zm1.ravel(), zm2.ravel()]).T
            z0 = np.repeat(z0u, rep0, axis=0)
            f0 = np.array([cls_data.sim_f(z, persis_info=persis_info) for z in z0])


        test = {
            "f": fg,
            "theta": tg,
            "xt": zg,
            "p": pg,
            "noise": ng,
            "p_prior": 1,
            "w": 1
        }

        # Methods to iterate over
        methods = ["ivar", "imse"]
        for mid, method in enumerate(methods):
            des_obj = sequential_design(cls_data)
            des_obj.build_design(
                z0=z0,
                f0=f0,
                T=maxiter,
                persis_info=persis_info,
                test=test,
                af=method,
                args={
                    "new": new,
                    "nL": 200,
                    "seed": s,
                    "method": method,
                    "t_grid": tg,
                    "neighbor": "LHS",
                    "extra_metric": False,
                },
            )

            visual_xt(cls_data, des_obj, z0u, rep0, tg, pg, ng, Xpl, Ypl, new)
