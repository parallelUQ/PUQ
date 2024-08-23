import numpy as np
from result_read import get_rep_data   
import matplotlib.pyplot as plt 
from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import find_threshold, plot_workers, plot_acc, plot_acqtime, plot_endtime, plot_errorend
### ### ### ### ### ### 

repno = 2
varlist = [ 1]
acclevel = 0.2
n = 2048
workers  = [16, 32, 64, 128, 256, 512]
batches  = [4, 8, 16]
simmeans = [2, 4, 16]
accparams = [[-1, 0.2], [-1, 0.21], [-1, 0.22]]
genparams = [[0.2, 0.2]]
              
result = []

for sid, sim_mean in enumerate(simmeans):
    for varid, var in enumerate(varlist):
        for id_b, b in enumerate(batches):
            res = []
            for r in range(repno):
                for id_w, w in enumerate(workers):
                    PM = performanceModel(worker=w, batch=b, n=n, n0=w)
                    PM.gen_gentime(genparams[0][0], 0.001, typeGen='constant')
                    PM.gen_simtime(sim_mean, sim_mean*var, 0.001, typeSim='normal', seed=r)
                    PM.gen_accuracy(accparams[id_b][0], accparams[id_b][1], typeAcc='exponential')
                    PM.simulate()
                    PM.summarize()
                    PM.complete(acclevel)

                    result.append({'r':r, 'b': b, 'var': var, 'w': w, 'simmean': sim_mean, 'res': PM})


clist = ['r', 'g', 'm', 'y', 'c']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['--', '-.', ':', '-.', ':'] 
labs = ['b=4', 'b=8', 'b=16']
ft = 18
for varid, var in enumerate(varlist): 
    fig, axes = plt.subplots(3, len(simmeans), figsize=(24, 18))     
    for sid, sim_mean in enumerate(simmeans):

        res_c = [res for res in result if ((res['simmean'] == sim_mean) & (res['var'] == var))]               
        for bid, b in enumerate(batches): 
            endtime = []
            idletime = []
            computetime = []
            for wid, w in enumerate(workers):
                endtime.append(np.mean([res['res'].complete_time for res in res_c if ((res['w'] == w) & (res['b'] == b))]))
                idletime.append(np.mean([res['res'].avg_idle_time for res in res_c if ((res['w'] == w) & (res['b'] == b))]))
                computetime.append(np.mean([res['res'].computing_hours for res in res_c if ((res['w'] == w) & (res['b'] == b))]))
                
            endtimescaled = [e/(endtime[0]) for e in endtime]
            axes[0, sid].plot(workers, endtimescaled,  marker=mlist[bid], markersize=10, linestyle=linelist[bid], linewidth=2.0, label=labs[bid], color=clist[bid])
            axes[1, sid].plot(workers, idletime,  marker=mlist[bid], markersize=10, linestyle=linelist[bid], linewidth=2.0, label=labs[bid], color=clist[bid])
            axes[2, sid].plot(workers, computetime,  marker=mlist[bid], markersize=10, linestyle=linelist[bid], linewidth=2.0, label=labs[bid], color=clist[bid])
            
            axes[0, sid].set_xscale('log')   
            axes[0, sid].set_yscale('log')   
            axes[0, sid].set_xticks(workers)
            axes[0, sid].set_xticklabels(workers, fontsize=14)
            axes[0, sid].tick_params(axis='both', which='major', labelsize=ft-2)
            #axes[0, sid].set_xlabel('# of workers', fontsize=16)
            
            axes[1, sid].set_xscale('log')   
            axes[1, sid].set_yscale('log')   
            axes[1, sid].set_xticks(workers)
            axes[1, sid].set_xticklabels(workers, fontsize=14)
            axes[1, sid].tick_params(axis='both', which='major', labelsize=ft-2)
            #axes[1, sid].set_xlabel('# of workers', fontsize=16)
            
            axes[2, sid].set_xscale('log')   
            axes[2, sid].set_yscale('log')   
            axes[2, sid].set_xticks(workers)
            axes[2, sid].set_xticklabels(workers, fontsize=14)
            axes[2, sid].tick_params(axis='both', which='major', labelsize=ft-2)
            axes[2, sid].set_xlabel('# of workers', fontsize=16)
            
            
        axes[0, sid].plot(workers, [1, 1/2, 1/4, 1/8, 1/16, 1/32], color='b', linewidth=4)
    axes[0, 0].set_ylabel('Wall-clock time (scaled)', fontsize=ft)
    axes[1, 0].set_ylabel('Idle time', fontsize=ft)
    axes[2, 0].set_ylabel('Computing hours', fontsize=ft)
    if varid == len(varlist)-1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', title_fontsize=25, 
                   bbox_to_anchor=(0.5, 0.05), 
                   ncol=4, 
                   prop={'size': 18},
                   fancybox=True, 
                   shadow=True)   
    plt.show()