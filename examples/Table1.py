
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
ex = 'sinfunc'
is_bias = False
if ex == 'pritam':
    n0, nf = 30, 180
    if is_bias:
        outs = 'pritam_bias'
        method = ['ceivarxbias', 'ceivarbias', 'lhs', 'rnd']

    else:
        outs = 'pritam'     
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
        
elif ex == 'sinfunc':
    n0, nf = 10, 100
    if is_bias:
        outs = 'sinf_bias'
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
    else:
        outs = 'sinf'    
        method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
        

labelsb = [r'$\mathcal{A}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$']
batch = 1
worker = 2
rep = 30
fonts = 18
metric = 'post'

path = '/Users/ozgesurer/Desktop/JQT_experiments/'
best_post, best_field = 0, 0
for mid, m in enumerate(method):
    avgPOST, sdPOST, avgPRED, sdPRED = plotresult(path, outs, ex, worker, batch, rep, m, n0=n0, nf=nf) 
    if metric == 'post':
        if m == 'lhs' or m == 'rnd':
            if best_post < avgPOST[-1]:
                best_post = avgPOST[-1]
    elif metric == 'pred':
        if m == 'lhs' or m == 'rnd':
            if best_field < avgPRED[-1]:
                best_field = avgPRED[-1]
                
if metric == 'post':
    if ex == 'pritam':
        best_post = 5 * 10**(-15)
        
for mid, m in enumerate(method):
    avgPOST, sdPOST, avgPRED, sdPRED = plotresult(path, outs, ex, worker, batch, rep, m, n0=n0, nf=nf)
    print(m)

    if metric == 'post':
        print(np.argmax((avgPOST <= best_post)) + 1)
    elif metric == 'pred':
        print(np.argmax((avgPRED <= best_field)) + 1)
