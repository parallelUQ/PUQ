
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd

def interval_score(theta, thetamle, tid):
    alpha = 0.1
    u = np.quantile(theta, 1-alpha/2)
    l = np.quantile(theta, alpha/2)

    is_l = 1 if thetamle < l else 0
    is_u = 1 if thetamle > u else 0
    

    if tid > 0:
        total_is = (u - l) + (2/alpha) * (l-thetamle) * (is_l) + (2/alpha) * (thetamle-u) * (is_u)
    else:
        total_is = (u - l)

    return total_is

def plotresult(path, out, ex_name, w, b, r0, rf, method, n0, nf):

    
    HDlist = []
    TVlist = []
    timelist = []
    for i in range(r0, rf):
        design_saved = read_output(path + out + '/', ex_name, method, w, b, i)

        TV       = design_saved._info['TV']
        HD       = design_saved._info['HD']

        TVlist.append(TV[n0:nf])
        HDlist.append(HD[n0:nf])
        
        theta = design_saved._info['theta']
        
        #import seaborn, pandas
        #seaborn.pairplot(pandas.DataFrame(theta))
        #plt.show()
    
    avgTV = np.mean(np.array(TVlist), 0)
    sdTV = np.std(np.array(TVlist), 0)
    avgHD = np.mean(np.array(HDlist), 0)
    sdHD = np.std(np.array(HDlist), 0)

    return avgHD, sdHD, avgTV, sdTV, theta


def FIG9(path, batch, worker, r0, rf, outs, ex, n0, nf):
    fonts = 16
    clist = ['b', 'dodgerblue', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
    mlist = ['P', 'p', '*', 'o', 's', 'h']
    linelist = ['-', '-', '--', '-.', ':', '-.', ':'] 
    labelsb = [r'$\mathcal{A}^y$', r'$\hat{\mathcal{A}}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$']
    method = ['ceivarx', 'ceivarxn', 'ceivar', 'lhs', 'rnd']

    labelsb = [r'$\mathcal{A}^y$', r'$\hat{\mathcal{A}}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$', r'$\mathcal{A}^{imspe}$', r'$\mathcal{A}^{var}$']
    method = ['ceivarx', 'ceivarxn', 'ceivar', 'lhs', 'rnd', 'imspe', 'maxvar']
    
    labelsb = [r'$\mathcal{A}^y$', r'$\hat{\mathcal{A}}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{var}$', r'$\mathcal{A}^{imspe}$']
    method = ['ceivarx', 'ceivarxn', 'ceivar', 'lhs', 'maxvar', 'imspe']

    #fig, axes = plt.subplots(1, 2, figsize=(20, 7)) 
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for metid, metric in enumerate(['HD', 'TV']):

        if metric == 'HD':
            axins = inset_axes(axes[metid], 2.7, 1.9 , loc=1, bbox_to_anchor=(0.5, 0.6), bbox_transform=axes[metid].transAxes)
            
        for mid, m in enumerate(method):
            avgPOST, sdPOST, avgPRED, sdPRED, _ = plotresult(path, outs, ex, worker, batch, r0, rf, m, n0=n0, nf=nf)
            if metric == 'TV':
                axes[metid].plot(np.arange(len(avgPRED)), avgPRED, label=labelsb[mid], color=clist[mid], linestyle=linelist[mid], linewidth=4)
            elif metric == 'HD':
                axes[metid].plot(np.arange(len(avgPOST)), avgPOST, label=labelsb[mid], color=clist[mid], linestyle=linelist[mid], linewidth=4)
                if m == 'lhs' or m == 'rnd' or m == 'imspe' or m == 'maxvar':
                    axins.plot(np.arange(50, len(avgPOST)), avgPOST[50:], color=clist[mid], linestyle=linelist[mid], linewidth=2)
    
        if metric == 'HD':
            mark_inset(axes[metid], axins, loc1=1, loc2=3, fc="none", ec="0.5")
            
        axes[metid].set_xlabel('# of simulation evaluations', fontsize=fonts) 
        axes[metid].set_yscale('log')
        if metric == 'TV':
            axes[metid].set_ylabel(r'${\rm MAD}^y$', fontsize=fonts) 
        elif metric == 'HD':
            axes[metid].set_ylabel(r'${\rm MAD}^p$', fontsize=fonts) 
        axes[metid].tick_params(axis='both', which='major', labelsize=fonts-2)
        axes[metid].legend(bbox_to_anchor=(0.9, -0.2), ncol=3, fontsize=fonts, handletextpad=0.1)
    plt.savefig('Figure11.jpg', format='jpeg', bbox_inches="tight", dpi=1000)
    plt.show()

      
def plot_IS(path, batch, worker, r0, rf, outs, ex, n0, nf):

    batch = 1
    worker = 2
    fonts = 18
    thetamle = 0.5
    n0 = 50
    nf = 200
    
    method = ['ceivarx', 'ceivarxn', 'ceivar', 'lhs', 'rnd']
    method = ['ceivarx', 'ceivarxn', 'ceivar', 'lhs', 'imspe', 'maxvar']
    result = []
    for mid, m in enumerate(method):
        for r in range(r0, rf):
            avgPOST, sdPOST, avgPRED, sdPRED, theta = plotresult(path, outs, ex, worker, batch, r, r+1, m, n0=n0, nf=nf)
 
            for th in range(0, 5):
                isval = interval_score(theta[n0:, th], thetamle, th)
                result.append({'method': m, 'score': isval, 'rep': r, 'th': th})
        
    result = pd.DataFrame(result)
    print(np.round(result.groupby(["method", "th"]).mean(), 2))
        
batch = 1
worker = 2
r0 = 0
rf = 30
n0, nf = 50, 200
#path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/newPUQcovid25/' 
#path = '/Users/ozgesurer/Desktop/covid19_bebop25/covidss/' 
# = '/Users/ozgesurer/Desktop/JQT_experiments/true_deneme/covid19_bebop25/all/' 
#path = '/Users/ozgesurer/Desktop/JQT_experiments/covid19_bebop68/all/' 
#path = '/Users/ozgesurer/Desktop/JQT_experiments/covid19_bebop25_unk/all/' 
path = '/Users/ozgesurer/Desktop/JQT_experiments/covid19_bebop25_response/' 
outs = 'covid19'
ex = 'covid19'

FIG9(path, batch, worker, r0, rf, outs, ex, n0, nf)

# plot_IS(path, batch, worker, r0, rf, outs, ex, n0, nf)