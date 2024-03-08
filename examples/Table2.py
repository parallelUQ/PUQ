import pandas as pd
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import seaborn as sns
from find_best_fit import find_bestfit_param

def plotresult(path, out, ex_name, w, b, rep, method):
    design_saved = read_output(path + out + '/', ex_name, method, w, b, rep)
    theta = design_saved._info['theta']
    return theta

def interval_score(theta, thetamle):
    alpha = 0.1
    u = np.quantile(theta, 1-alpha/2)
    l = np.quantile(theta, alpha/2)
    is_l = 1 if thetamle < l else 0
    is_u = 1 if thetamle > u else 0

    total_is = (u - l) + (2/alpha) * (l-thetamle) * (is_l) + (2/alpha) * (thetamle-u) * (is_u)

    return total_is


clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 

# choose either 'pritam' or 'sinfunc'
ex = 'pritam'
is_bias = True
if ex == 'pritam':
    n0, nf = 30, 80
    if is_bias:
        outs = 'pritam_bias'
        method = ['ceivarxbias', 'ceivarbias', 'lhs', 'rnd']
    else:
        outs = 'pritam'     
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
        
elif ex == 'sinfunc':
    n0, nf = 10, 30
    if is_bias:
        outs = 'sinf_bias'
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
    else:
        outs = 'sinf'    
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
        

thetamle = find_bestfit_param(ex)

labelsb = [r'$\mathcal{A}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$']
batch = 1
worker = 2
rep = 5
fonts = 18
path = '/Users/ozgesurer/Desktop/JQT_experiments/'
# Score boxplots
r0 = 0
rf = 30
result = []
for mid, m in enumerate(method):
    total_inc = 0

    for r in range(r0, rf):
        theta = plotresult(path, outs, ex, worker, batch, r, m)

        if ex == 'sinfunc':
            isval = interval_score(theta[n0:, 1], thetamle)
        else:
            isval = interval_score(theta[n0:, 2], thetamle)
        
        result.append({'method': m, 'score': isval, 'rep': r})

 
result = pd.DataFrame(result)
print(np.round(result.groupby(["method"]).mean(), 2))
# print(np.round(result.groupby(["method"]).median(), 2))
       
# ax = sns.boxplot(data=result, x='method', y='score', fill=False, palette=['b', 'r', 'g', 'purple'])
# ax.set_xticks(ax.get_xticks())
# ax.set_xticklabels(labelsb)
# ax.tick_params(axis='both', which='major', labelsize=fonts-3)
# ax.set_ylabel("Interval Score", fontsize=fonts)
# ax.set_xlabel("Acquisition Functions", fontsize=fonts)
# plt.show()
