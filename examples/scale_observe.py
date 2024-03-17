from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

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

worker = 2
batch = 1
r0 = 1
rf = 4
n0 = 30
nf = 180
path = '/Users/ozgesurer/Desktop/JQT_experiments/highdim_scale/'

mesh = [250, 500, 1000, 2000]
fig, axes = plt.subplots(1, 1, figsize=(5, 4)) 
for m in mesh:
    example_name = 'ceivar' + str(m)
    avgPOST, sdPOST, avgPRED, sdPRED, theta = plotresult(path, 'highdim', worker, batch, r0, rf, example_name, n0=n0, nf=nf)
    axes.plot(avgPOST, label=str(m))
axes.set_yscale('log')   
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.74, 0.0), ncol=5, fontsize=15, handletextpad=0.1)
plt.show()

res = []
res.append({'time': 1702.573, 'method': 250, 'r': 1})
res.append({'time': 2118.497, 'method': 500, 'r': 1})
res.append({'time': 3104.459, 'method': 1000, 'r': 1})
res.append({'time': 5780.347, 'method': 2000, 'r': 1})
res.append({'time': 1693.164, 'method': 250, 'r': 2})
res.append({'time': 2089.0, 'method': 500, 'r': 2})
res.append({'time': 3144.845, 'method': 1000, 'r': 2})
res.append({'time': 5834.193, 'method': 2000, 'r': 2})
res.append({'time': 1688.932, 'method': 250, 'r': 3})
res.append({'time': 2082.294, 'method': 500, 'r': 3})
res.append({'time': 3126.343, 'method': 1000, 'r': 3})
res.append({'time': 5798.706, 'method': 2000, 'r': 3})

import pandas as pd
import matplotlib.ticker as ticker
res = pd.DataFrame(res)
summ = np.round(res.groupby(["method"]).mean(), 2)

fig, axes = plt.subplots(1, 1, figsize=(5, 4)) 
axes.scatter([250, 500, 1000, 2000], summ['time'], color='b', s=50)
axes.plot(np.array([250, 500, 1000, 2000]), summ['time'], color='b')
axes.set_xlabel('Size of the reference set', fontsize=18)  
axes.set_ylabel('Time (seconds)', fontsize=18)  
axes.set_yscale('log')  
axes.set_xscale('log')
axes.set_xticks([250, 500, 1000, 2000])
axes.set_xticklabels([250, 500, 1000, 2000], fontsize=15)
axes.set_yticks(np.round(np.array(summ['time'])))
axes.set_yticklabels(np.round(np.array(summ['time'])), fontsize=15)
axes.minorticks_off()
plt.show()