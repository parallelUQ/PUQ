import numpy as np
from result_read import get_rep_data   
import matplotlib.pyplot as plt 
from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import find_threshold, plot_workers, plot_acc, plot_acqtime, plot_endtime, plot_errorend
### ### ### ### ### ### 

repno = 10
varlist = [1]
acclevel = 0.2
n = 2560
worker = 256
n0 = worker
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
simmeans = [4, 16, 64]
clist = ['b', 'r', 'g', 'm', 'y', 'c']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 
lab = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16', 'b=32', 'b=64', 'b=128', 'b=256']
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
genparams = [1, 0.5, 0.25]
result = []


for sid, sim_mean in enumerate(simmeans):
    for varid, var in enumerate(varlist):
        for aid, acc in enumerate(accparams):
            res = []
            for r in range(repno):
    
                for id_b, b in enumerate(batches):
                    PM = performanceModel(worker=worker, batch=b, n=n, n0=n0)
                    PM.gen_gentime(genparams[aid], 0.001, typeGen='constant')
                    PM.gen_simtime(simmeans[sid], simmeans[sid], 0.001, typeSim='normal', seed=r)
                    PM.gen_accuracy(acc[0], acc[1] + id_b*0.01, typeAcc='exponential')

                    PM.simulate()
                    PM.summarize()
                    PM.complete(acclevel)

                    #plot_workers(PM, PM.job_list, PM.stage_list)
                    result.append({'r':r, 'b': b, 'var': var, 'acc': aid, 'simmean': simmeans[sid], 'res': PM})

labs = [r'$\mathcal{A}_1$', r'$\mathcal{A}_2$', r'$\mathcal{A}_3$']
for varid, var in enumerate(varlist): 
    fig, axes = plt.subplots(1, len(simmeans), figsize=(15, 4))     
    for sid, sim_mean in enumerate(simmeans):

        res_c = [res for res in result if ((res['simmean'] == sim_mean) & (res['var'] == var))]               
        for aid, acc in enumerate(accparams): 
            endtime = []
            
            for bid, b in enumerate(batches):
                endtime.append(np.mean([res['res'].complete_time for res in res_c if ((res['b'] == b) & (res['acc'] == aid))]))

            axes[sid].plot(batches, endtime,  marker=mlist[aid], markersize=10, linestyle=linelist[aid], linewidth=2.0, label=labs[aid], color=clist[aid])
            axes[sid].set_xscale('log')   
            axes[sid].set_yscale('log')   
            axes[sid].set_xticks(batches)
            axes[sid].set_xticklabels(batches, fontsize=14)
            axes[sid].tick_params(axis='both', which='major', labelsize=14)
            axes[sid].set_xlabel('b', fontsize=16)
    axes[0].set_ylabel('Wall-clock time', fontsize=16)
    if varid == len(varlist)-1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', title_fontsize=25, 
                   bbox_to_anchor=(0.5, -0.1), 
                   ncol=4, 
                   prop={'size': 18},
                   fancybox=True, 
                   shadow=True)   
    plt.show()