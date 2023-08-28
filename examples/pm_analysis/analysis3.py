import numpy as np
import matplotlib.pyplot as plt
from PUQ.performance import performanceModel

# # # # # # # # # # # #
# # # # HEAT MAP # # # #
# # # # # # # # # # # #

worker_size = 64

no_jobs = 1024
batch_size = [1, 4, 16, 64]

acq = [10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10**1, 10**2, 10**3, 10**4, 10**5]
sim = [10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10**1, 10**2, 10**3, 10**4, 10**5]

#acq = [2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 1, 2**1, 2**2, 2**3, 2**4, 2**5]
#sim = [2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 1, 2**1, 2**2, 2**3, 2**4, 2**5]

result = []
for b in batch_size:
    for a in acq:
        for s in sim:
            PM = performanceModel(worker=worker_size, n0=0, batch=b, n=no_jobs)
            
            PM.gen_gentime(a/b, typeGen='constant')
            PM.gen_simtime(s, typeSim='constant')
            PM.gen_accuracy(-1, 0.2, typeAcc='exponential')
            
            PM.simulate()
            
            PM.summarize()
            result.append({'a':a, 'b':b, 's':s, 'r':PM.end_time})

        
import pandas as pd
fig, axes = plt.subplots(3, 3, figsize=(15,15)) 

result = pd.DataFrame(result)
ft = 18

for i in range(0, len(batch_size)): # row
    for j in range(i+1, len(batch_size)): # column
            
            print(batch_size[i])
            print(batch_size[j])
            e1 = result[result['b'] == batch_size[i]]['r'] 
            e2 = result[result['b'] == batch_size[j]]['r'] 
            print(np.array(e1)/np.array(e2))
            pcm = axes[j-1, i].scatter(result[result['b'] == 1]['a'], 
                                 result[result['b'] == 1]['s'], 
                                 c=np.array(e1)/np.array(e2), 
                                 zorder=1)

            axes[j-1, i].set_yscale('log')
            axes[j-1, i].set_xscale('log')
            cbar = fig.colorbar(pcm, ax= axes[j-1, i])
            cbar.ax.tick_params(labelsize=ft) 
            axes[j-1, i].tick_params(axis='both', labelsize=ft)

for i in range(len(batch_size)-1):
    axes[i, 0].set_ylabel(r'$b$:' + str(batch_size[i+1]), fontsize=ft)

for i in range(len(batch_size)-1):
    axes[len(batch_size)-2, i].set_xlabel(r'$b$:' + str(batch_size[i]), fontsize=ft)
    
for i in range(len(batch_size)-1):
    for j in range(i+1, len(batch_size)-1):
        axes[i, j].axis('off')

# fig.title('|Worker|:', worker_size)
#plt.suptitle('|Worker|:' + str(worker_size), fontsize=ft)
fig.supylabel('Simulation Time', fontsize=ft)
fig.supxlabel('Acquisition Time', fontsize=ft)

plt.tight_layout()