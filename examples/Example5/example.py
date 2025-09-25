import numpy as np
from utils import fig6
from utils_sample import test_data_gen
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
from PUQ.designmethods.sequential_1d_stochastic import sequential_design
from test_functions import unimodalx, bimodalx, braninx
    
T, s = 100, 1
# Methods to iterate over
dict_meth = {
    "method": ["ivar"],
    "horizon": [
        {"method": "target", "h0": 0, "target_ratio": 0.2},
    ],
    "labels": ["target"],
}
fig, ax = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)

if __name__ == "__main__":
    
    for eid, example in enumerate(["unimodalx", "bimodalx", "braninx"]):

        cex = eval(example)()
        cex.realdata(x=np.array([0.5])[:, None], seed=None)
    
        tg, fg, pg, zg, ng, t_s, p_s, w_s, f_s, n_s, Xpl, Ypl = test_data_gen(cex, sample=True)
        
        tdat = {
            "f": f_s,
            "theta": t_s,
            "xt": None,
            "p": p_s,
            "noise": n_s,
            "w": w_s,
            "p_prior": 1,
        }
        
        gdat = {"f": fg, "theta": tg, "xt": zg, "p": pg, "noise": ng, "X": Xpl, "Y": Ypl}

        
        # Set random stream for initial design
        persis_info = {"rand_stream": np.random.default_rng(s)}
    
        # Initial sample
        n0, rep0 = 30, 5
        sampling = LHS(xlimits=cex.zlim, random_state=int(s))
        z0u = sampling(n0)
        z0 = np.repeat(z0u, rep0, axis=0)
        f0 = np.array(
            [cex.sim_f(z0[i, :], persis_info=persis_info) for i in range(n0 * rep0)]
        )
    
        for mid, method in enumerate(dict_meth["method"]):
            print(method)
            # Set random stream for initial design
            persis_info = {"rand_stream": np.random.default_rng(s)}
    
            # Set random stream for initial design
            des_obj = sequential_design(cex)
            des_obj.build_design(
                z0=z0,
                f0=f0,
                T=T,
                persis_info=persis_info,
                test=tdat,
                af="lookahead",
                args={
                    "horizon": dict_meth["horizon"][mid],
                    "long": False,
                    "nL": 300,
                    "seed": s,
                    "method": method,
                    "t_grid": None,
                    "integral": "importance",
                },
            )
    

            fig6(
                des_obj,
                z0[:, 1:3],
                rep0,
                gdat["X"],
                gdat["Y"],
                gdat["p"],
                gdat["noise"],
                ax[eid],
                {},
                fig,
            )

plt.savefig("ex5.png", format="jpeg", bbox_inches="tight", dpi=500)
plt.show()