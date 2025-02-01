import numpy as np
from PUQ.prior import prior_dist
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.stats as sps
from PUQ.design import designer
from PUQ.utils import parse_arguments, save_output
from smt.sampling_methods import LHS
from sir_funcs import SEIRDS
import seaborn as sns
import pandas as pd

args = parse_arguments()

# # # # # 
args.minibatch = 16
args.seedmin = 1
args.seedmax = 2
# # # # # 

workers = args.minibatch + 1
n0 = 50
rep0 = 2
rho = 1/2
batch = args.minibatch
maxiter = 16


# # Create test data
nt = 2500
nrep = 1000

cls_func = eval('SEIRDS')()
sampling = LHS(xlimits=cls_func.thetalimits, random_state=100)
theta_test = sampling(nt)
d = cls_func.d
f_test = np.zeros((nt, d))
f_var = np.zeros((nt, d))

persis_info = {}
persis_info['rand_stream'] = np.random.default_rng(100)

for thid, th in enumerate(theta_test):
    IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
    f_test[thid, :] = np.mean(IrIdRD, axis=0)
    f_var[thid, :]  = np.var(IrIdRD, axis=0)
    plt.plot(np.arange(0, d), f_test[thid, :])
IrIdRDtrueall = cls_func.sim_f(thetas=cls_func.theta_true, return_all=True, repl=nrep, persis_info=persis_info)
IrIdRDtrue = np.mean(IrIdRDtrueall, axis=0)
#print(np.round(np.var(IrIdRDtrueall, axis=0), 3))
plt.scatter(np.arange(0, d), IrIdRDtrue, zorder=2)
plt.show()




if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):
        
        cls_func = eval('SEIRDS')()
        cls_func.realdata(seed=s)

        for i in range(5):
            rnd = sps.norm(f_test[-1, :][i], np.sqrt(cls_func.obsvar[i, i]))
            print(rnd.pdf(cls_func.real_data[0, i]))
    
        p_test = np.zeros(nt)
        for thid, th in enumerate(theta_test):
            rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
            p_test[thid] = rnd.pdf(cls_func.real_data)
        
        test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}
        
        sns.set(font_scale=2.0)
        g = sns.pairplot(pd.DataFrame(theta_test[p_test > 0.00001, ]))
        for i in range(0, 7):
            for j in range(0, 7):
                g.axes[i, j].set_xlim((0, 1))
            g.axes[i, i].axvline(x = cls_func.theta_true[i])
        plt.show()
     
        plt.plot(np.arange(0, len(IrIdRDtrue)), f_test[np.argmax(p_test), :], color='red')
        plt.plot(np.arange(0, len(IrIdRDtrue)), f_test[-1, :], color='blue')
        plt.scatter(np.arange(0, len(IrIdRDtrue)), cls_func.real_data)  
        plt.show()
        # Set a uniform prior
        prior_func = prior_dist(dist="uniform")(
            a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
        )

        plt.plot(np.arange(0, len(IrIdRDtrue)), IrIdRDtrue)
        plt.scatter(np.arange(0, len(IrIdRDtrue)), cls_func.real_data)  
        plt.show()

        # Initial sample
        sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
        theta0   = sampling(n0)
        theta0   = np.repeat(theta0, rep0, axis=0)
        f0       = np.zeros((cls_func.d, n0*rep0))
        for i in range(0, n0*rep0):
            f0[:, i] = cls_func.sim_f(theta0[i, :], persis_info=persis_info)
            
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
                "des_init":{'seed':s, 'n0':n0, 'rep0':rep0, 'theta':theta0, 'f':f0},
                "alloc_settings":{'method':'ivar', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':True, 'nL': 500, 'impute_str': 'update'}
            },
        )

        save_output(al_ivar, cls_func.data_name, 'ivar', workers, batch, s)

        sns.set(font_scale=2.0)
        g = sns.pairplot(pd.DataFrame(al_ivar._info['theta'][(n0*rep0):,]))
        for i in range(0, 7):
            for j in range(0, 7):
                g.axes[i, j].set_xlim((0, 1))
            g.axes[i, i].axvline(x = cls_func.theta_true[i])
        plt.show()
        
        plt.plot(al_ivar._info['f'][:, (n0*rep0):], zorder=1)
        plt.scatter(np.arange(0, cls_func.d), cls_func.real_data.flatten(), zorder=2)
        plt.show()
        
        # al_imse = designer(
        #     data_cls=cls_func,
        #     method="p_sto_bseq",
        #     acquisition="simse",
        #     args={
        #         "prior": prior_func,
        #         "data_test": test_data,
        #         "max_iter": maxiter,
        #         "nworkers": workers,
        #         "batch_size": batch,
        #         "des_init":{'seed':s, 'n0':n0, 'rep0':rep0, 'theta':theta0, 'f':f0},
        #         "alloc_settings":{'method':'imse', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
        #         "pc_settings":{'standardize':True, 'latent':False},
        #         "des_settings":{'is_exploit':True, 'is_explore':True, 'nL': 500, 'impute_str': 'update'}
        #     },
        # )
        
        # save_output(al_imse, cls_func.data_name, 'imse', workers, batch, s)
        # sns.pairplot(pd.DataFrame(al_imse._info['theta'][(n0*rep0):,]))
        # plt.show()
        
        # rholhs = 1/4
        # sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
        # theta_lhs = sampling(int(maxiter*rholhs))

        # al_unif = designer(
        #     data_cls=cls_func,
        #     method="p_sto_bseq",
        #     acquisition=None,
        #     args={
        #         "prior": prior_func,
        #         "data_test": test_data,
        #         "max_iter": maxiter,
        #         "nworkers": workers,
        #         "batch_size": batch,
        #         "des_init":{'seed':s, 'theta':theta0, 'f':f0},
        #         "des_add":{'theta':theta_lhs},
        #         "alloc_settings":{'use_Ki':True, 'rho': rholhs, 'theta':None, 'a0':None, 'gen':False},
        #         "pc_settings":{'standardize':True, 'latent':False},
        #         "des_settings":{'is_exploit':True, 'is_explore':True}
        #     },
        # )
        
        # save_output(al_unif, cls_func.data_name, 'unif', workers, batch, s)
        
        # plt.plot(al_ivar._info['TV'], color='red')    
        # plt.plot(al_imse._info['TV'], color='blue')  
        # plt.plot(al_unif._info['TV'], color='green')  
        # plt.yscale('log')
        # plt.show()