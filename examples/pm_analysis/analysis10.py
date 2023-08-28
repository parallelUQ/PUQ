import numpy as np
from result_read import get_rep_data   
import matplotlib.pyplot as plt 
from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import find_threshold, plot_workers, plot_acc, plot_acqtime, plot_endtime, plot_errorend
### ### ### ### ### ### 

repno = 30
varlist = [0.1, 1]
acclevel = 0.2
scale_list = [1.2, 1.4, 1.6, 1.8, 2]
n = 2048
workers  = [16, 32, 64, 128, 256, 512]
batches  = [4, 8, 16]
simmeans = [1, 2, 4, 16]
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
accparams = [[-1, 0.2]]
genparams = [[0.2, 0.2], [0.08, 0.02], [0.02, 0.002]]
genparams = [[0.2, 0.2]]
              
result = []

for sid, sim_mean in enumerate(simmeans):
    for varid, var in enumerate(varlist):
        for id_b, b in enumerate(batches):
            res = []
            for r in range(repno):
                for id_w, w in enumerate(workers):
                    if b == 1:
                        cons = 1
                    else:
                        cons             = np.arange(scale_list[id_b], 1, -(scale_list[id_b]-1)/n)[0:n] 
                        
                    PM = performanceModel(worker=w, batch=b, n=n, n0=w)
                    PM.gen_gentime(genparams[0][0], typeGen='constant')
                    PM.gen_simtime(sim_mean, sim_mean*var, typeSim='normal', seed=r)
                    PM.gen_accuracy(accparams[0][0], accparams[0][1], typeAcc='exponential')
                    PM.gen_accuracy(cons*PM.acc, typeAcc='batched')
                    PM.simulate()
                    PM.summarize()
                    PM.complete(acclevel)
                    PM.simulate()
                    PM.summarize()
                    result.append({'r':r, 'b': b, 'var': var, 'w': w, 'simmean': sim_mean, 'res': PM})


clist = ['r', 'g', 'm', 'y', 'c']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['--', '-.', ':', '-.', ':'] 
labs = ['b=4', 'b=8', 'b=16']
for varid, var in enumerate(varlist): 
    fig, axes = plt.subplots(1, len(simmeans), figsize=(20, 4))     
    for sid, sim_mean in enumerate(simmeans):

        res_c = [res for res in result if ((res['simmean'] == sim_mean) & (res['var'] == var))]               
        for bid, b in enumerate(batches): 
            endtime = []
            
            for wid, w in enumerate(workers):
                endtime.append(np.mean([res['res'].complete_time for res in res_c if ((res['w'] == w) & (res['b'] == b))]))

            endtimescaled = [e/(endtime[0]) for e in endtime]
            #endtimescaled = [endtime[0]*i for i in [1, 1/2, 1/4, 1/8, 1/16]]
            axes[sid].plot(workers, endtimescaled,  marker=mlist[bid], markersize=10, linestyle=linelist[bid], linewidth=2.0, label=labs[bid], color=clist[bid])
            #axes[sid].plot(workers, endtimescaled, linewidth=2.0, color=clist[bid], alpha=0.2)
            axes[sid].set_xscale('log')   
            #axes[sid].set_yscale('log')   
            axes[sid].set_xticks(workers)
            axes[sid].set_xticklabels(workers, fontsize=14)
            axes[sid].tick_params(axis='both', which='major', labelsize=14)
            axes[sid].set_xlabel('# of workers', fontsize=16)
        axes[sid].plot(workers, [1, 1/2, 1/4, 1/8, 1/16, 1/32], color='b', linewidth=4)
    axes[0].set_ylabel('Completion time (scaled)', fontsize=16)
    if varid == len(varlist)-1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', title_fontsize=25, 
                   bbox_to_anchor=(0.5, -0.1), 
                   ncol=4, 
                   prop={'size': 18},
                   fancybox=True, 
                   shadow=True)   
    plt.show()