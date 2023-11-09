
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def plotresult(path, out, ex_name, w, b, rep, method, n0, nf):

    
    HDlist = []
    TVlist = []
    timelist = []
    for i in range(0, rep):
        design_saved = read_output(path + out + '/', ex_name, method, w, b, i)

        TV       = design_saved._info['TV']
        HD       = design_saved._info['HD']

        TVlist.append(TV[n0:nf])
        HDlist.append(HD[n0:nf])

    avgTV = np.mean(np.array(TVlist), 0)
    sdTV = np.std(np.array(TVlist), 0)
    avgHD = np.mean(np.array(HDlist), 0)
    sdHD = np.std(np.array(HDlist), 0)

    return avgHD, sdHD, avgTV, sdTV


def FIG9(path, batch, worker, rep, outs, ex, n0, nf):
    fonts = 15
    clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
    mlist = ['P', 'p', '*', 'o', 's', 'h']
    linelist = ['-', '--', '-.', ':', '-.', ':'] 
    labelsb = [r'$\mathcal{A}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$']

    method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
    for metric in ['TV', 'HD']:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5)) 

        if metric == 'HD':
            axins = inset_axes(axes, 2.3, 1.5 , loc=1, bbox_to_anchor=(0.5, 0.6), bbox_transform=axes.transAxes)
            
        for mid, m in enumerate(method):
            avgPOST, sdPOST, avgPRED, sdPRED = plotresult(path, outs, ex, worker, batch, rep, m, n0=n0, nf=nf)
            if metric == 'TV':
                axes.plot(np.arange(len(avgPRED)), avgPRED, label=labelsb[mid], color=clist[mid], linestyle=linelist[mid], linewidth=4)
                #plt.fill_between(np.arange(len(avgPRED)), avgPRED-1.96*sdPRED/rep, avgPRED+1.96*sdPRED/rep, color=clist[mid], alpha=0.1)
            elif metric == 'HD':
                axes.plot(np.arange(len(avgPOST)), avgPOST, label=labelsb[mid], color=clist[mid], linestyle=linelist[mid], linewidth=4)
                #plt.fill_between(np.arange(len(avgPOST)), avgPOST-1.96*sdPOST/rep, avgPOST+1.96*sdPOST/rep, color=clist[mid], alpha=0.1)  
                if m == 'lhs' or m == 'rnd':
                    axins.plot(np.arange(50, len(avgPOST)), avgPOST[50:], color=clist[mid], linestyle=linelist[mid], linewidth=2)
    
        if metric == 'HD':
            mark_inset(axes, axins, loc1=1, loc2=3, fc="none", ec="0.5")
            
        axes.set_xlabel('# of simulation evaluations', fontsize=fonts) 
        axes.set_yscale('log')
        if metric == 'TV':
            axes.set_ylabel(r'${\rm MAD}^y$', fontsize=fonts) 
        elif metric == 'HD':
            axes.set_ylabel(r'${\rm MAD}^p$', fontsize=fonts) 
        axes.tick_params(axis='both', which='major', labelsize=fonts-2)
        axes.legend(bbox_to_anchor=(0.9, -0.2), ncol=4, fontsize=fonts, handletextpad=0.1)
        if metric == 'TV':
            plt.savefig("Figure9_pred.png", bbox_inches="tight")
        elif metric == 'HD':
            plt.savefig("Figure9_post.png", bbox_inches="tight")
        plt.show()
      
    
batch = 1
worker = 2
rep = 30
n0, nf = 50, 200
path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/newPUQcovid25/' 
outs = 'covid19'
ex = 'covid19'

FIG9(path, batch, worker, rep, outs, ex, n0, nf)