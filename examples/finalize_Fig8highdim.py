from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from collections import OrderedDict


def interval_score(theta, thetamle, xid, tid):
    alpha = 0.1
    u = np.quantile(theta, 1-alpha/2)
    l = np.quantile(theta, alpha/2)

    is_l = 1 if thetamle < l else 0
    is_u = 1 if thetamle > u else 0
    
    if xid == 0:
        if tid > 1:
            total_is = (u - l) + (2/alpha) * (l-thetamle) * (is_l) + (2/alpha) * (thetamle-u) * (is_u)
        else:
            total_is = (u - l)
    elif xid == 1:
        if tid > 5:
            total_is = (u - l) + (2/alpha) * (l-thetamle) * (is_l) + (2/alpha) * (thetamle-u) * (is_u)
        else:
            total_is = (u - l)
    elif xid == 2:    
        if tid > 9:
            total_is = (u - l) + (2/alpha) * (l-thetamle) * (is_l) + (2/alpha) * (thetamle-u) * (is_u)
        else:
            total_is = (u - l)        
        
    return total_is

def plotresult(path, ex_name, w, b, r0, rf, method, n0, nf):

    HDlist = []
    TVlist = []
    timelist = []
    for i in range(r0, rf):
        design_saved = read_output(path, ex_name, method, w, b, i)

        TV       = design_saved._info['TV']
        HD       = design_saved._info['HD']

        TVlist.append(TV[n0:nf])
        HDlist.append(HD[n0:nf])
        
        theta = design_saved._info['theta']
        f = design_saved._info['f']

    avgTV = np.mean(np.array(TVlist), 0)
    sdTV = np.std(np.array(TVlist), 0)
    avgHD = np.mean(np.array(HDlist), 0)
    sdHD = np.std(np.array(HDlist), 0)

    return avgHD, sdHD, avgTV, sdTV, theta


def plot_aggregated(example_name='sinfunc', is_bias=False, r0=0, rf=30,
                    clist=None, mlist=None, linelist=None, 
                    labelsb=None, path=None, method=None):
    
    
    batch = 1
    worker = 2
    fonts = 18

    rep = rf - r0
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 9)) 
    plt.rcParams["figure.autolayout"] = True
    
    for exid, ex in enumerate(['x2/', 'x6/', 'x10/']):
        print(ex)
    #for exid, ex in enumerate(['x6/']):
        if exid == 0:
            n0 = 50
            nf = 200
        else:
            n0 = 30
            nf = 180
        
        for metid, metric in enumerate(['HD']):
            #for mid, m in enumerate(method):
            for mid, (name, linestyle) in enumerate(linestyles.items()):

                avgPOST, sdPOST, avgPRED, sdPRED, theta = plotresult(path+ex, example_name, worker, batch, r0, rf, method[mid], n0=n0, nf=nf)

                if metric == 'TV':
                    axes[metid, exid].plot(np.arange(len(avgPRED)), avgPRED, label=labelsb[mid], color=clist[mid], ls=linestyle, linewidth=4)
                    #plt.fill_between(np.arange(len(avgPRED)), avgPRED-1.96*sdPRED/rep, avgPRED+1.96*sdPRED/rep, color=clist[mid], alpha=0.1)
                elif metric == 'HD':
                    axes[metid, exid].plot(np.arange(len(avgPOST)), avgPOST, label=labelsb[mid], color=clist[mid], ls=linestyle, linewidth=4)
                    #plt.fill_between(np.arange(len(avgPOST)), avgPOST-1.96*sdPOST/rep, avgPOST+1.96*sdPOST/rep, color=clist[mid], alpha=0.1)  
            axes[metid, exid].set_yscale('log')
            if metid == 1:
                axes[metid, exid].set_xlabel('# of simulation evaluations', fontsize=fonts) 
        
            if metric == 'TV':
                if exid == 0:
                    axes[metid, exid].set_ylabel(r'${\rm MAD}^y$', fontsize=fonts) 
            elif metric == 'HD':
                if exid == 0:
                    axes[metid, exid].set_ylabel(r'${\rm MAD}^p$', fontsize=fonts) 
            axes[metid, exid].tick_params(axis='both', which='major', labelsize=fonts-5)
            
    handles, labels = axes[metid, exid].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.74, 0.0), ncol=5, fontsize=fonts, handletextpad=0.1)
    plt.show()


def plot_IS(example_name='sinfunc', is_bias=False, r0=0, rf=30, 
            clist=None, mlist=None, linelist=None, 
            labelsb=None, path=None, method=None):

    batch = 1
    worker = 2
    fonts = 18
    thetamle = 0.5
    
    for exid, ex in enumerate(['x2/', 'x6/', 'x10/']):
        if exid == 0:
            n0 = 50
            nf = 200
        else:
            n0 = 30
            nf = 180
            
        result = []
        for mid, m in enumerate(method):
            for r in range(r0, rf):
                avgPOST, sdPOST, avgPRED, sdPRED, theta = plotresult(path+ex, example_name, worker, batch, r, r+1, m, n0=n0, nf=nf)
     
                for th in range(0, 12):
                    isval = interval_score(theta[n0:, th], thetamle, exid, th)
                    result.append({'method': m, 'score': isval, 'rep': r, 'th': th})
            
        result = pd.DataFrame(result)
        print(np.round(result.groupby(["method", "th"]).mean(), 2))

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('densely dashed',      (0, (5, 1))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),
     ('densely dotted',      (0, (1, 1))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
mlist = ['P', 'p', '*', 'o', 's', 'h']
labelsb = [r'$\mathcal{A}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$',  r'$\mathcal{A}^{var}$',  r'$\mathcal{A}^{imspe}$']
method = ['ceivarx', 'ceivar', 'lhs', 'maxvar', 'imspe']

path = '/Users/ozgesurer/Desktop/JQT_experiments/highdim_ex/'
plot_aggregated(example_name='highdim', 
                is_bias=False, 
                r0=1, rf=8, 
                clist=clist, 
                mlist=mlist, 
                linelist=linestyles, 
                labelsb=labelsb, 
                path=path, 
                method=method)

# plot_IS(example_name='highdim', 
#         is_bias=False, 
#         r0=1, rf=8, 
#         clist=clist, 
#         mlist=mlist, 
#         linelist=linestyles, 
#         labelsb=labelsb, 
#         path=path, 
#         method=method)