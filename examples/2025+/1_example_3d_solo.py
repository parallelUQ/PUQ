import numpy as np
import matplotlib.pyplot as plt
from test_functions import unimodalx, bimodalx, braninx
from smt.sampling_methods import LHS
from PUQ.designmethods.sequential import sequential_design
from utils import fig6, create_entry, save_output
from paper_figs import lineplot, boxplot
from utils_sample import test_data_gen
import pandas as pd


explore = True
maxiter = 50
seedmin, seedmax = 0, 1

dfl = []

example = "braninx"

if __name__ == "__main__":

    cex = eval(example)()
    cex.realdata(x=np.array([0.5])[:, None], seed=None)
    
    tg, fg, pg, zg, ng, t_s, p_s, w_s, f_s, n_s, Xpl, Ypl = test_data_gen(cex, sample=True)

    for s in np.arange(seedmin, seedmax):

        # Set random stream for initial design
        persis_info = {'rand_stream': np.random.default_rng(s)}
        
        if explore:
            # Initial sample
            n0, rep0 = 30, 5
            sampling = LHS(xlimits=cex.zlim, random_state=int(s))
            z0u = sampling(n0)
            z0 = np.repeat(z0u, rep0, axis=0)
            f0 = np.array([cex.sim_f(z0[i, :], persis_info=persis_info) for i in range(n0 * rep0)])
        else:
            # Initial sample
            grid_size, rep0 = 10, 2
            z1 = np.linspace(cex.zlim[1][0], cex.zlim[1][1], grid_size)
            z2 = np.linspace(cex.zlim[1][0], cex.zlim[1][1], grid_size)
            zm1, zm2 = np.meshgrid(z1, z2)
            z0u = np.vstack([zm1.ravel(), zm2.ravel()]).T
            z0 = np.repeat(z0u, rep0, axis=0)
            x_tiled = np.tile(np.array([[0.40], [0.50]]), (z0.shape[0], 1))
            t_repeated = np.repeat(z0, np.array([[0.40], [0.50]]).shape[0], axis=0)
            z0 = np.hstack([x_tiled, t_repeated])
            f0 = np.array([cex.sim_f(z, persis_info=persis_info) for z in z0])

        test = {'f': f_s, 
                'theta': t_s, 
                'xt': None, 
                'p': p_s, 
                'noise': n_s, 
                "w": w_s,
                "p_prior": 1}
        

        # Methods to iterate over
        methods = ["ivar", "imse", "var"]

        labs = ["ivar", "imse", "var"]
        for mid, method in enumerate(methods):

            des_obj = sequential_design(cex)
            des_obj.build_design(z0=z0, 
                                 f0=f0, 
                                 T=maxiter, 
                                 persis_info=persis_info, 
                                 test=test, 
                                 af=method,
                                 args={"new":explore, 
                                       "nL":200, 
                                       "seed":s, 
                                       "t_grid":None,
                                       "integral":"importance"})
     
            dfl.extend(create_entry(des_obj, cex.data_name, method, s, labs[mid]))
            
            fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
            fig6(des_obj, z0[:, 1:3], rep0, Xpl, Ypl, pg, ng, ax, {}, fig, exp=explore)
            plt.show()

    df = pd.DataFrame(dfl)
    df2 = df.groupby(["s", "h", "example", "method"])["new"].mean().mul(100).reset_index()
    df2.rename(columns={"new": "percent"}, inplace=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    lineplot(df, [example], metric="MAD", hue="method", ax=ax[0])
    boxplot(df=df2, ax=ax[1], x="h", hue="method")
    plt.show()