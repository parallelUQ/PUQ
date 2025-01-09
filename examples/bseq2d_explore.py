import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments, save_output
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen, twoD, heatmap
import pandas as pd
from PUQ.design import designer
from smt.sampling_methods import LHS

args = parse_arguments()

# # # # # 
args.minibatch = 16
args.funcname = 'banana'
args.seedmin = 0
args.seedmax = 5

# # # # # 

workers = args.minibatch + 1
n0 = 15
rep0 = 2
nmesh = 50
rho = 1/2
batch = args.minibatch
maxiter = 256
dfl, dfr = [], []
if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):
        
        cls_func = eval(args.funcname)()
        cls_func.realdata(seed=s)

        theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
        test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}

        heatmap(cls_func)
    
        # Set a uniform prior
        prior_func = prior_dist(dist="uniform")(
            a=cls_func.thetalimits[:, 0], b=cls_func.thetalimits[:, 1]
        )
        
        # Set random stream for initial design
        persis_info = {'rand_stream': np.random.default_rng(s)}
        
        # Initial sample
        sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
        theta0 = sampling(n0)
        theta0 = np.repeat(theta0, rep0, axis=0)
        f0     = np.zeros((cls_func.d, n0*rep0))
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
                "des_init":{'seed':s, 'theta':theta0, 'f':f0},
                "alloc_settings":{'method':'ivar', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':True, 'nL':200, 'impute_str': 'update'}
            },
        )
        
        save_output(al_ivar, cls_func.data_name, 'ivar', workers, batch, s)
        twoD(al_ivar, Xpl, Ypl, p_test, nmesh)
        
        dfl.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'ivar', 'example':args.funcname} for t, MAD in enumerate(al_ivar._info['TViter'])])
        dfr.extend([{'rep':s, 'batch':batch, 'worker':workers, 'method':'ivar', 'example':args.funcname,
                    'iter_explore': al_ivar._info['iter_explore'], 'iter_exploit': al_ivar._info['iter_exploit']}])
        
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
        #         "des_init":{'seed':s, 'theta':theta0, 'f':f0},
        #         "alloc_settings":{'method':'imse', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
        #         "pc_settings":{'standardize':True, 'latent':False},
        #         "des_settings":{'is_exploit':True, 'is_explore':True, 'nL':200, 'impute_str': 'update'}
        #     },
        # )
        
        # save_output(al_imse, cls_func.data_name, 'imse', workers, batch, s)
        # twoD(al_imse, Xpl, Ypl, p_test, nmesh)
        
#         dfl.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'imse', 'example':args.funcname} for t, MAD in enumerate(al_imse._info['TViter'])])
#         dfr.extend([{'rep':s, 'batch':batch, 'worker':workers, 'method':'imse', 'example':args.funcname,
#                     'iter_explore': al_imse._info['iter_explore'], 'iter_exploit': al_imse._info['iter_exploit']}])

#         rholhs = 1/4
#         sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(s))
#         theta_lhs = sampling(int(maxiter*rholhs))

#         al_unif = designer(
#             data_cls=cls_func,
#             method="p_sto_bseq",
#             acquisition=None,
#             args={
#                 "prior": prior_func,
#                 "data_test": test_data,
#                 "max_iter": maxiter,
#                 "nworkers": workers,
#                 "batch_size": batch,
#                 "des_init":{'seed':s, 'theta':theta0, 'f':f0},
#                 "des_add":{'theta':theta_lhs},
#                 "alloc_settings":{'use_Ki':True, 'rho':rholhs, 'theta':None, 'a0':None, 'gen':False},
#                 "pc_settings":{'standardize':True, 'latent':False},
#                 "des_settings":{'is_exploit':True, 'is_explore':True}
#             },
#         )
        
#         save_output(al_unif, cls_func.data_name, 'unif', workers, batch, s)
#         twoD(al_unif, Xpl, Ypl, p_test, nmesh)
        
#         dfl.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'unif', 'example':args.funcname} for t, MAD in enumerate(al_unif._info['TViter'])])
 

        al_var = designer(
            data_cls=cls_func,
            method="p_sto_bseq",
            acquisition="var",
            args={
                "prior": prior_func,
                "data_test": test_data,
                "max_iter": maxiter,
                "nworkers": workers,
                "batch_size": batch,
                "des_init":{'seed':s, 'theta':theta0, 'f':f0},
                "alloc_settings":{'method':'ivar', 'use_Ki':True, 'rho': rho, 'theta':None, 'a0':None, 'gen':False},
                "pc_settings":{'standardize':True, 'latent':False},
                "des_settings":{'is_exploit':True, 'is_explore':True, 'nL':200, 'impute_str': 'update'}
            },
        )
        
        save_output(al_var, cls_func.data_name, 'var', workers, batch, s)
        twoD(al_var, Xpl, Ypl, p_test, nmesh)
        
        dfl.extend([{'MAD':MAD, 't':t, 'rep':s, 'batch':batch, 'worker':workers, 'method':'var', 'example':args.funcname} for t, MAD in enumerate(al_var._info['TViter'])])
        dfr.extend([{'rep':s, 'batch':batch, 'worker':workers, 'method':'ivar', 'example':args.funcname,
                    'iter_explore': al_var._info['iter_explore'], 'iter_exploit': al_var._info['iter_exploit']}])
        
from summary import lineplot, exp_ratio
df = pd.DataFrame(dfl)
lineplot(df, examples=[args.funcname], batches=[batch])
#df = pd.DataFrame(dfr)
#exp_ratio(df, examples=[args.funcname], methods=['ivar', 'imse'], batches=[batch], ntotals=[maxiter])

# from PUQ.utils import parse_arguments, read_output
# import matplotlib.pyplot as plt
# folderpath = '/Users/ozgesurer/Desktop/stochastic/'
# b = 8
# r = 15
# m = 'ivar'
# example = 'banana' 
# path = folderpath + '1' + '_' + example + '_' + 'explore' + '/'  + str(b) + '/'

# desobj = read_output(path, example, m, b+1, b, r)
# reps0 = desobj._info['reps0']
# theta0 = desobj._info['theta0']
# theta = desobj._info['theta']     
# f = desobj._info['f']         

# plt.scatter(f, al_ivar._info['f'])   
# plt.show()      

# from summary import visual_theta
# visual_theta(example=example,
#               method=m, 
#               batch=8, 
#               rep0=15, repf=16, 
#               initial=30, 
#               folderpath='/Users/ozgesurer/Desktop/stochastic/',
#               ids='1',
#               ee='explore')

# df = pd.DataFrame(dfl)
# plt.plot(np.arange(0, len(df.loc[df['method'] == 'ivar']['MAD'])), df.loc[df['method'] == 'ivar']['MAD'], color='red')
# plt.plot(np.arange(0, len(df.loc[df['method'] == 'ivar']['MAD'])), df.loc[df['method'] == 'imse']['MAD'])

