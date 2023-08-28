from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import plot_acc, plot_acqtime, plot_endtime, plot_errorend
import matplotlib.pyplot as plt 
import numpy as np

timeparams = [0.008, 0.006, 0.004]
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
acclevel = 0.1
result = []
n = 4096
worker = 1
n0 = 0
for mid, m in enumerate(timeparams):
    PM = performanceModel(worker=1, batch=1, n=n, n0=n0)
    PM.gen_gentime(timeparams[mid], typeGen='constant')
    PM.gen_simtime(0.0001, 0.0001, typeSim='normal')
    PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
    PM.simulate()
    PM.summarize()
    PM.complete(acclevel)
    result.append(PM)

fig, axes = plt.subplots(2, 2, figsize=(22, 20)) 
plot_acc(axes[0, 0], n, acclevel, result, labellist=['M1', 'M2', 'M3'], logscale=False, fontsize=25)
plot_acqtime(axes[0, 1], n, acclevel, result, labellist=['M1', 'M2', 'M3'], logscale=False, fontsize=25)
plot_endtime(axes[1, 0], n, acclevel, result, labellist=['M1', 'M2', 'M3'], worker=worker, logscale=False, fontsize=25)
plot_errorend(axes[1, 1], n, acclevel, result, labellist=['M1', 'M2', 'M3'], worker=worker, logscale=False, fontsize=25)
plt.show()


acclevel = 0.1
n = 2048
worker = 64
n0 = 64
timeparams = [[1, 1], [0.8, 0.8], [0.6, 0.6]]
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
result = []
for mid, m in enumerate(timeparams):
    
    PM = performanceModel(worker=worker, batch=1, n=n, n0=n0)
    PM.gen_gentime(timeparams[mid][0], timeparams[mid][1], typeGen='linear')
    PM.gen_simtime(25, 0.0001, typeSim='normal')
    PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
    PM.simulate()
    PM.summarize()
    PM.complete(acclevel)
    result.append(PM)
    
fig, axes = plt.subplots(2, 2, figsize=(22, 20)) 
plot_acc(axes[0, 0], n, acclevel, result, labellist=['M1', 'M2', 'M3'], logscale=False, fontsize=25)
plot_acqtime(axes[0, 1], n, acclevel, result, labellist=['M1', 'M2', 'M3'], logscale=False, fontsize=25)
plot_endtime(axes[1, 0], n, acclevel, result, labellist=['M1', 'M2', 'M3'], worker=worker, logscale=False, fontsize=25)
plot_errorend(axes[1, 1], n, acclevel, result, labellist=['M1', 'M2', 'M3'], worker=worker, logscale=False, fontsize=25)
plt.show()


# Batch version
n = 2048
timeparams = [1]
batches = [1, 8, 64]
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
result = []
worker = 64
n0 = 64
for mid, m in enumerate(batches):
    PM = performanceModel(worker=worker, batch=batches[mid], n=n, n0=n0)
    PM.gen_gentime(timeparams[0], typeGen='constant')
    PM.gen_simtime(25, 0.0001, typeSim='normal')
    PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
    PM.simulate()
    PM.summarize()
    PM.complete(acclevel)
    result.append(PM)


fig, axes = plt.subplots(2, 2, figsize=(22, 20)) 
plot_acc(axes[0, 0], n, acclevel, result, labellist=['b=1', 'b=4', 'b=16'], logscale=False, fontsize=25)
plot_acqtime(axes[0, 1], n, acclevel, result, labellist=['b=1', 'b=4', 'b=16'], logscale=False, fontsize=25)
plot_endtime(axes[1, 0], n, acclevel, result, labellist=['b=1', 'b=4', 'b=16'], worker=worker, logscale=False, fontsize=25)
plot_errorend(axes[1, 1], n, acclevel, result, labellist=['b=1', 'b=4', 'b=16'], worker=worker, logscale=False, fontsize=25)
plt.show()


### 
timeparams = [[1, 1], [0.8, 0.8], [0.6, 0.6]]
accparams = [[-1, 0.05], [-1, 0.06], [-1, 0.07]]
batches = [1, 2, 4, 8, 16, 32, 64]
result = []
acclevel=0.1
scale_list = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
n = 2048
n0 = 64
worker = 64


for mid, m in enumerate(timeparams):
    result = []

    for bid, b in enumerate(batches):

        if b == 1:
            cons = 1
        else:
            cons             = np.arange(scale_list[bid], 1, -(scale_list[bid]-1)/n)[0:n] 
                
        PM = performanceModel(worker=worker, batch=b, n=n, n0=n0)
        PM.gen_gentime(timeparams[mid][0], timeparams[mid][1], typeGen='linear')
        PM.gen_simtime(10, 0.1, typeSim='normal')
        PM.gen_accuracy(accparams[mid][0], accparams[mid][1], typeAcc='exponential')
        PM.acc = cons*PM.acc 
        PM.simulate()
        PM.summarize()
        PM.complete(acclevel=acclevel)
        result.append(PM)

        
    labs = [str(b) for b in batches]
    fig, axes = plt.subplots(2, 2, figsize=(22, 20)) 
    plot_acc(axes[0, 0], n, acclevel, result, labellist=labs, logscale=False, fontsize=25)
    plot_acqtime(axes[0, 1], n, acclevel, result, labellist=labs, logscale=False, fontsize=25)
    plot_endtime(axes[1, 0], n, acclevel, result, labellist=labs, worker=worker, logscale=False, fontsize=25)
    plot_errorend(axes[1, 1], n, acclevel, result, labellist=labs, worker=worker, logscale=False, fontsize=25)
    plt.show()