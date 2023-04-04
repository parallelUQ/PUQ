import numpy as np
from result_read import get_rep_data   
import matplotlib.pyplot as plt 
from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import find_threshold
### ### ### ### ### ### 
n = 2048
rep = 1
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

PM = performanceModel(worker=1, batch=1, n=n)
PM.gen_accuracy(x_a, y_a, xtest_a, typeAcc='regress')
fitted_acc = np.exp(PM.acc)
### ### ### ### ### ### 

repno = 1
varlist = [1, 5, 25]
acc  = 0.00001

scale_list = [1, 1.12, 1.25, 1.37, 1.5, 1.62, 1.75, 1.87, 2]
worker  = 256
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
time_param = 5#[[0, 5], [0, 4.5], [0, 4], [0, 3.5], [0, 3], [0, 2.5], [0, 2], [0, 1.5], [0.0, 1]]
simmeans = [10, 20, 40]
acqscale = [0.25, 0.5, 1, 2]
clist = ['b', 'r', 'g', 'm', 'y', 'c']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 


fig, axes = plt.subplots(len(simmeans), len(acqscale), figsize=(20, 20)) 


for scaleid, scale in enumerate(acqscale):
    for varid, var in enumerate(varlist):
        res = []
        for sid, sim_mean in enumerate(simmeans):
            result = []
            for r in range(repno):

                for id_b, b in enumerate(batches):
                    if b == 1:
                        cons = 1
                    else:
                        cons             = np.arange(scale_list[id_b], 1, -(scale_list[id_b]-1)/n)[0:n] 
                        
                    PM = performanceModel(worker=512, batch=b, n=n)
  
                    ## ##
                    PM.gen_gentime((time_param/b)*scale, typeGen='constant')
                    ## ##  
                    
                    ## ##
                    PM.gen_simtime(sim_mean, sim_mean*var, typeSim='normal')
                    ## ##
                    
                    ## ##
                    minl = np.min(avgae)
                    maxl = np.max(avgae)
                    lnew = [(litem - 0)/(maxl - 0) for litem in avgae]
    
                    x_a      = np.log(np.arange(1, len(lnew)+1)) 
                    y_a      = np.log(lnew)
                    xtest_a  = np.log(np.arange(1, n+1))
                    PM.gen_accuracy(x_a, y_a, xtest_a, typeAcc='regress')
                    PM.acc = cons*np.exp(PM.acc)
                    ## ##
                    
                    PM.simulate()
                    
                    PM.summarize()
                    #result.append(PM)
                    endjob_r         = [job['end'] for job in PM.job_list]  
                    result.append({'r': r, 'b': b, 'accuracy': PM.acc, 'acq_time': PM.gentime, 'end_job': endjob_r})
            res.append(result)
    
                
        for sid, s in enumerate(simmeans):
            endtime = []
            for bid, b in enumerate(batches):
                avgend = np.mean([re['end_job'] for re in res[sid] if re['b'] == b ], axis=0)
                avgacc = np.mean([re['accuracy'] for re in res[sid] if re['b'] == b ], axis=0)
                thr, thrid = find_threshold(acc, avgacc)
                endtime.append(avgend[thrid])
            axes[varid, scaleid].plot(batches, endtime,  marker=mlist[sid], linestyle=linelist[sid], label=str(s))
            axes[varid, scaleid].set_xscale('log')   
            axes[varid, scaleid].set_yscale('log')   
            axes[varid, scaleid].set_xticks(batches)
            axes[varid, scaleid].set_xticklabels(batches, fontsize=14)
            axes[varid, scaleid].tick_params(axis='both', which='major', labelsize=14)
            #axes[scaleid, sid].set_title(title + "%1.0e"%stime, fontsize=16)
            axes[varid, scaleid].set_xlabel('b', fontsize=16)
#axes[0, 0].set_title('s=1', fontsize=16)
#axes[0, 1].set_title('s=2', fontsize=16)
#axes[0, 2].set_title('s=4', fontsize=16)
#axes[0, 3].set_title('s=8', fontsize=16)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', title_fontsize=18,
           bbox_to_anchor=(0.5, 0.05), 
           ncol=4, 
           prop={'size': 16},
           fancybox=True, 
           shadow=True)   
plt.show()


