
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np

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



clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 

# choose either 'pritam' or 'sinfunc'
ex = 'pritam'
is_bias = False
if ex == 'pritam':
    path = '/Users/ozgesurer/Desktop/JQT_Experiments/n0_pritam/' 
    ns = [15, 30, 60, 120]
    #ns = [30, 60]
    labelsb = [r'$15$', r'$30$', r'$60$', r'$120$']
    #labelsb = [r'$30$', r'$60$']
    n0, nf = 30, 180
    if is_bias:
        outs = 'pritam_bias'
        method = ['ceivarxbias', 'ceivarbias', 'lhs', 'rnd']

    else:
        outs = 'pritam'     
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
        
elif ex == 'sinfunc':
    path = '/Users/ozgesurer/Desktop/JQT_Experiments/n0_sinf/' 
    ns = [5, 10, 20, 40]
    labelsb = [r'$5$', r'$10$', r'$20$', r'$40$']
    n0, nf = 10, 100
    if is_bias:
        outs = 'sinf_bias'
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
    else:
        outs = 'sinf'    
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
        

batch = 1
worker = 2
rep = 2
fonts = 18
m = 'ceivar'

for metric in ['TV', 'HD']:
    fig, axes = plt.subplots(1, 1, figsize=(6, 5)) 
    plt.rcParams["figure.autolayout"] = True
    for nid, n0 in enumerate(ns):
        outs = str(n0)
        avgPOST, sdPOST, avgPRED, sdPRED = plotresult(path, outs, ex, worker, batch, rep, m, n0=n0, nf=nf)
        if metric == 'TV':
            axes.plot(np.arange(n0, nf), avgPRED, label=labelsb[nid], color=clist[nid], linestyle=linelist[nid], linewidth=4)
            #plt.fill_between(np.arange(n0, nf), avgPRED-1.96*sdPRED/rep, avgPRED+1.96*sdPRED/rep, color=clist[nid], alpha=0.1)
        elif metric == 'HD':
            axes.plot(np.arange(n0, nf), avgPOST, label=labelsb[nid], color=clist[nid], linestyle=linelist[nid], linewidth=4)
            #plt.fill_between(np.arange(n0, nf), avgPOST-1.96*sdPOST/rep, avgPOST+1.96*sdPOST/rep, color=clist[nid], alpha=0.1)  
    axes.set_yscale('log')
    #axes.set_xscale('log')
    axes.set_xlabel('# of simulation evaluations', fontsize=fonts) 

    if metric == 'TV':
        axes.set_ylabel(r'${\rm MAD}^y$', fontsize=fonts) 
    elif metric == 'HD':
        axes.set_ylabel(r'${\rm MAD}^p$', fontsize=fonts) 
    axes.tick_params(axis='both', which='major', labelsize=fonts-5)
    
    axes.legend(bbox_to_anchor=(1.1, -0.2), ncol=4, fontsize=fonts, handletextpad=0.1)
    plt.show()