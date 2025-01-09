import numpy as np
from PUQ.prior import prior_dist
from PUQ.utils import parse_arguments
from PUQ.design import designer
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen, twoD, heatmap
from smt.sampling_methods import LHS
from scipy.stats import linregress
import matplotlib.pyplot as plt

args = parse_arguments()

# # # # # 
args.funcname = 'bimodal'
args.seedmin = 0
args.seedmax = 10
# # # # # 

n0 = 15
rep0 = 2
nmesh = 50
rho = 1/2

maxiter = 320
time_dict = []
if __name__ == "__main__":
    
    for batch in [8, 16, 32, 64]:
        workers = batch + 1
        for s in np.arange(args.seedmin, args.seedmax):
            
            cls_func = eval(args.funcname)()
            cls_func.realdata(seed=s)
    
            theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
            test_data = {"theta": theta_test, "f": f_test, "p": p_test, "p_prior": 1}
    
            # heatmap(cls_func)
        
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
                    "des_settings":{'is_exploit':False, 'is_explore':True, 'nL':200, 'impute_str': 'update'}
                },
            )
            
            # twoD(al_ivar, Xpl, Ypl, p_test, nmesh)
            
            time = [t for t in al_ivar._info['time']]
            
            time_dict.append({'time': time, 'rep': s, 'batch': batch})
        
    # Process data to compute average arrays for each batch
    batch_averages = {}
    for entry in time_dict:
        batch = entry['batch']
        time_array = np.array(entry['time'])
        if batch not in batch_averages:
            batch_averages[batch] = []
        batch_averages[batch].append(time_array)

    # Compute average for each batch
    result = {batch: np.mean(np.array(times), axis=0) for batch, times in batch_averages.items()}

    cs = ['blue', 'red', 'green', 'cyan']
    ms = ['^', '*', 'o', 'D']
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ft = 20
    idc = 0

    for batch, avg_times in result.items():
        xlm = np.arange(1, len(avg_times) + 1)
        ylm = avg_times
        # Fit a linear regression model to the data
        slope, intercept, r_value, p_value, std_err = linregress(xlm, ylm)
        trend_line = slope * xlm + intercept
   
        ax.scatter(xlm, ylm, marker=ms[idc], color=cs[idc], label=str(batch), s=100)
        #ax.plot(xlm, trend_line, color=cs[idc], linestyle='--', linewidth=2)
        
        # Calculate total time and annotate
        total_time = np.sum(avg_times)
        ax.annotate(f"{total_time:.0f}",
                    xy=(xlm[-1], ylm[-1]),  # Position the annotation at the last point
                    xytext=(xlm[-1] + 0.5, ylm[-1]),  # Offset the text a bit
                    fontsize=ft, color=cs[idc])

        idc += 1
    ax.set_xlabel('t', fontsize=ft)
    ax.set_ylabel('Time (sec.)', fontsize=ft)   
    ax.tick_params(axis="both", labelsize=ft)
    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3),
              fancybox = True, shadow = True, ncol = 4, fontsize=ft-5)
    plt.savefig('time_explore.png', bbox_inches='tight')
    plt.show()

