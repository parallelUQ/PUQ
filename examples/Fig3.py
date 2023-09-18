from plotutils import plotresult, plotparams
import matplotlib.pyplot as plt 
import numpy as np

# Generates Fig 3, 4, 11, 12
clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 

labelsb = ['EI', 'EIVAR', 'HYBRID', 'RND']
method = ['ei', 'eivar', 'hybrid_ei', 'rnd']
batch = 1
worker = 2
rep = 30
fonts = 22

path = '/Users/ozgesurer/Desktop/sh_files/'
figs = [['himmelblau', 'holder', 'easom'], ['sphere', 'matyas', 'ackley']]
for example_name in figs:
    for metric in ['AE', 'MAD']:
        fig, axes = plt.subplots(1, 3, figsize=(22, 5)) 
        for exid, ex in enumerate(example_name):
            for mid, m in enumerate(method):
                out = ex + '_' + m
                avgAE, avgtime, avgTV = plotresult(path, out, ex, worker, batch, rep, m, n0=0, nf=1000)
                #print(avgAE[0:10])
                if metric == 'AE':
                    axes[exid].plot(np.arange(len(avgAE)), avgAE, label=labelsb[mid], color=clist[mid], linestyle=linelist[mid])
                else:
                    axes[exid].plot(np.arange(len(avgTV)), avgTV, label=labelsb[mid], color=clist[mid], linestyle=linelist[mid])
            axes[exid].set_yscale('log')
            #axes[exid].set_xscale('log')
            axes[exid].set_xlabel('# of parameters', fontsize=fonts) 
            if exid == 0:
                if metric == 'AE':
                    axes[exid].set_ylabel(r'$\delta$', fontsize=fonts) 
                else:
                    axes[exid].set_ylabel(r'MAD', fontsize=fonts) 
            axes[exid].tick_params(axis='both', which='major', labelsize=fonts-5)
        
        axes[1].legend(bbox_to_anchor=(1.3, -0.2), ncol=4, fontsize=fonts)
        plt.show()
