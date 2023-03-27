from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import plot_accuracy
import matplotlib.pyplot as plt
import numpy as np

# Batch version
timeparams = [[1, 1], [0.8, 0.8], [0.6, 0.6]]
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
result = []
for mid, m in enumerate(timeparams):
    
    PM = performanceModel(worker=64, batch=1, n=2048)
    
    PM.gen_gentime(timeparams[mid][0], timeparams[mid][1], typeGen='linear')

    PM.gen_simtime(25, 0.0001, typeSim='normal')
    PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
    
    PM.simulate()
    
    PM.summarize()
    result.append(PM)

plot_accuracy(result, n=2048, acclevel=0.1, labellist=['b=1', 'b=4', 'b=16'], logscale=False)

# Batch version
timeparams = [[1, 1], [0.8, 0.8], [0.6, 0.6]]
accparams = [[-1, 0.05], [-1, 0.06], [-1, 0.07]]
batches = [1, 2, 4, 8, 16, 32, 64]
result = []
acclevel=0.1
scale_list = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
n = 2048

fig, axes = plt.subplots(1, 1, figsize=(5, 5)) 
for mid, m in enumerate(timeparams):
    result = []
    resultPM = []
    for bid, b in enumerate(batches):

        if b == 1:
            cons = 1
        else:
            cons             = np.arange(scale_list[bid], 1, -(scale_list[bid]-1)/n)[0:n] 
                
        PM = performanceModel(worker=64, batch=b, n=n)
        
        PM.gen_gentime(timeparams[mid][0], timeparams[mid][1], typeGen='linear')
    
        PM.gen_simtime(10, 0.1, typeSim='normal')
        PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
        PM.acc = cons*PM.acc 
        
        PM.simulate()
        
        PM.summarize()
        
        endtime = PM.complete(acclevel=acclevel)
        
        result.append(endtime)
        resultPM.append(PM)
    plot_accuracy(resultPM, n=n, acclevel=acclevel, labellist=[str(b) for b in batches], logscale=True, worker=64)

    
    axes.plot(batches, result, label='M'+str(mid))
axes.set_xticks(batches)
axes.set_xticklabels(batches, fontsize=14)
axes.tick_params(axis='both', which='major', labelsize=14)
axes.set_xscale('log')   
axes.set_yscale('log') 

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', title_fontsize=18,
           bbox_to_anchor=(0.5, 0.05), 
           ncol=4, 
           prop={'size': 16},
           fancybox=True, 
           shadow=True)   
plt.show()      