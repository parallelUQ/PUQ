import numpy as np
from result_read import get_rep_data   
import matplotlib.pyplot as plt 
from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import find_threshold, plot_workers, plot_acc, plot_acqtime, plot_endtime, plot_errorend
### ### ### ### ### ### 
n = 2048
rep = 30
example = 'himmelblau_ex'
s = 'himmelblau'
lab = 'hybrid_ei_c1000/'

path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/'
filename = path + 'performanceAnalytics/new_fun_all/new_examples/' + example + '/' + lab
avgae, avgtime = get_rep_data(s, 2, 1, rep, filename, 'hybrid_ei')

minl = np.min(avgae)
maxl = np.max(avgae)
lnew = [(litem - 0)/(maxl - 0) for litem in avgae]

x_a  = np.log(np.arange(1, len(lnew)+1)) 
y_a  = np.log(lnew)
xtest_a  = np.log(np.arange(1, n+1))

PM = performanceModel(worker=1, batch=1, n=n, n0=0)
PM.gen_accuracy(x_a, y_a, xtest_a, typeAcc='regress')
fitted_acc = np.exp(PM.acc)
### ### ### ### ### ### 

repno = 1
varlist = [0.1, 10]
acclevel  = 0.00001

#scale_list = [1, 1.12, 1.25, 1.37, 1.5, 1.62, 1.75, 1.87, 2]
scale_list = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6]
#scale_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

worker  = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
simmeans = [0.1, 1, 10]
acqscale = [0.01, 0.1, 1, 10]

clist = ['b', 'r', 'g', 'm', 'y', 'c']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 
thrlist = []

lab = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16', 'b=32', 'b=64', 'b=128', 'b=256']

                    
result = []
for scaleid, scale in enumerate(acqscale):
    for varid, var in enumerate(varlist):
        for sid, sim_mean in enumerate(simmeans):
            res = []
            for r in range(repno):

                for id_b, b in enumerate(batches):
                    if b == 1:
                        cons = 1
                    else:
                        cons             = np.arange(scale_list[id_b], 1, -(scale_list[id_b]-1)/n)[0:n] 
                        
                    PM = performanceModel(worker=worker, batch=b, n=n, n0=worker)
  
                    ## ##
                    PM.gen_gentime(1*scale, 1*scale, typeGen='linear')
                    ## ##  
                    
                    ## ##
                    PM.gen_simtime(sim_mean, sim_mean*var, typeSim='normal')
                    PM.gen_accuracy(cons*fitted_acc, typeAcc='batched')
                    PM.simulate()
                    PM.summarize()
                    PM.complete(acclevel)
                    PM.simulate()
                    PM.summarize()
                    
                    #plot_workers(PM, PM.job_list, PM.stage_list)
                    result.append({'r':r, 'b': b, 'scale': scale, 'var': var, 'simmean': sim_mean, 'res': PM})
        
for varid, var in enumerate(varlist): 
    fig, axes = plt.subplots(1, len(acqscale), figsize=(20, 4))     
    for scaleid, scale in enumerate(acqscale):

        res_c = [res for res in result if ((res['scale'] == scale) & (res['var'] == var))]               
        for sid, s in enumerate(simmeans):
            endtime = []
            
            for bid, b in enumerate(batches):
                endtime.append(np.mean([res['res'].complete_time for res in res_c if ((res['b'] == b) & (res['simmean'] == s))]))

            axes[scaleid].plot(batches, endtime,  marker=mlist[sid], markersize=10, linestyle=linelist[sid], linewidth=2.0, label=str(s), color=clist[sid])
            axes[scaleid].set_xscale('log')   
            axes[scaleid].set_yscale('log')   
            axes[scaleid].set_xticks(batches)
            axes[scaleid].set_xticklabels(batches, fontsize=14)
            axes[scaleid].tick_params(axis='both', which='major', labelsize=14)
            axes[scaleid].set_xlabel('b', fontsize=16)
    axes[0].set_ylabel('Completion time', fontsize=16)
    if varid == len(varlist)-1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', title_fontsize=25, title = r'$\mathbb{E}[s]$', 
                   bbox_to_anchor=(0.5, 0.01), 
                   ncol=4, 
                   prop={'size': 18},
                   fancybox=True, 
                   shadow=True)   
    plt.show()


