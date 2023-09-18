from plotutils import plotresult, plotparams
import matplotlib.pyplot as plt 
import numpy as np


clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 

labelsb = ['b=1', 'b=5', 'b=25', 'b=125']
method = ['hybrid_ei']
batch_sizes = [1, 5, 25, 125]
worker = 126
rep = 30
fonts = 22
nf = 1000

path = '/Users/ozgesurer/Desktop/batch_mode/'
figs = [['sphere', 'matyas', 'ackley'], ['himmelblau', 'holder', 'easom']]
for example_name in figs:
    for metric in ['AE', 'MAD']:
        fig, axes = plt.subplots(1, 3, figsize=(22, 5)) 
        for exid, ex in enumerate(example_name):
            for bid, b in enumerate(batch_sizes):
                for mid, m in enumerate(method):
                    out = ex + '_' + m + '_' + 'b' + str(b) + '_' + 'w125'
                    avgAE, avgtime, avgTV = plotresult(path, out, ex, worker, b, rep, m, n0=123, nf=nf)
                    if metric == 'AE':
                        axes[exid].plot(np.arange(len(avgAE)), avgAE, label=labelsb[bid], color=clist[bid], linestyle=linelist[bid])
                    else:
                        axes[exid].plot(np.arange(len(avgTV)), avgTV, label=labelsb[bid], color=clist[bid], linestyle=linelist[bid])
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

show = False
if show:    
    clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
    mlist = ['P', 'p', '*', 'o', 's', 'h']
    linelist = ['-', '--', '-.', ':', '-.', ':'] 

    labelsb = ['b=1', 'b=32', 'b=64', 'b=128']
    m = 'hybrid_ei'
    ex = 'easom'
    worker = 126
    rep = 28
    fonts = 22
    from test_funcs import easom
    cls_data = easom()
    thetalim = cls_data.thetalimits
    
    path = '/Users/ozgesurer/Desktop/sh_files/batch_mode/'
    b = 125
    out = ex + '_' + m + '_' + 'b' + str(b) + '_' + 'w125'
    plotparams(path, out, ex, worker, b, rep, m, n0=123, nf=nf, thetalim=thetalim)
    
