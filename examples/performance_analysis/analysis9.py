import numpy as np
from result_read import get_rep_data   
import matplotlib.pyplot as plt 
from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import find_threshold, plot_workers, plot_acc, plot_acqtime, plot_endtime, plot_errorend
### ### ### ### ### ### 

repno = 30
varlist = [1]
acclevel = 0.2

#scale_list = [1, 1.12, 1.25, 1.37, 1.5, 1.62, 1.75, 1.87, 2]
scale_list = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
#scale_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
n = 2048
worker  = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
simmeans = [2, 4, 16]

clist = ['b', 'r', 'g', 'm', 'y', 'c']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 
lab = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16', 'b=32', 'b=64', 'b=128', 'b=256']

accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
genparams = [[0.2, 0.2], [0.02, 0.02], [0.002, 0.002]]
      
genparams = [[0.2, 0.2], [0.08, 0.02], [0.02, 0.002]]

result = []
sim_mean = 1
var = 1
for mid, m in enumerate(genparams):
    PM = performanceModel(worker=256, batch=1, n=n)
    PM.gen_gentime(genparams[mid][0], typeGen='constant')
    PM.gen_simtime(sim_mean, sim_mean*var, typeSim='normal')
    
    PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
    PM.simulate()
    PM.summarize()
    PM.complete(acclevel)
    result.append(PM)

fig, axes = plt.subplots(2, 2, figsize=(22, 20)) 
plot_acc(axes[0, 0], n, acclevel, result, labellist=['M1', 'M2', 'M3'], logscale=False, fontsize=25)
plot_acqtime(axes[0, 1], n, acclevel, result, labellist=['M1', 'M2', 'M3'], logscale=False, fontsize=25)
plot_endtime(axes[1, 0], n, acclevel, result, labellist=['M1', 'M2', 'M3'], worker=worker, logscale=False, fontsize=25)
plot_errorend(axes[1, 1], n, acclevel, result, labellist=['M1', 'M2', 'M3'], worker=worker, logscale=False, fontsize=25)




              
result = []

#for scaleid, scale in enumerate(acqscale):
for sid, sim_mean in enumerate(simmeans):
    for varid, var in enumerate(varlist):
        for aid, acc in enumerate(accparams):#change to alg
            res = []
            for r in range(repno):
    
                for id_b, b in enumerate(batches):
                    if b == 1:
                        cons = 1
                    else:
                        cons             = np.arange(scale_list[id_b], 1, -(scale_list[id_b]-1)/n)[0:n] 
                        
                    PM = performanceModel(worker=worker, batch=b, n=n)
                    #PM.gen_gentime(genparams[aid][0], genparams[aid][1], typeGen='linear')
                    PM.gen_gentime(genparams[aid][0], typeGen='constant')
                    PM.gen_simtime(sim_mean, sim_mean*var, typeSim='normal')
                    PM.gen_accuracy(acc[0], acc[1], typeAcc='exponential')
                    PM.gen_accuracy(cons*PM.acc, typeAcc='batched')

                    PM.simulate()
                    PM.summarize()
                    PM.complete(acclevel)
                    PM.simulate()
                    PM.summarize()
                    
                    #plot_workers(PM, PM.job_list, PM.stage_list)
                    result.append({'r':r, 'b': b, 'var': var, 'acc': aid, 'simmean': sim_mean, 'res': PM})
                #plt.show()
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
    axes[0].set_ylabel('Completion time', fontsize=16)
    if varid == len(varlist)-1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', title_fontsize=25, 
                   bbox_to_anchor=(0.5, -0.1), 
                   ncol=4, 
                   prop={'size': 18},
                   fancybox=True, 
                   shadow=True)   
    plt.show()


