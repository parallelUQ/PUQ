from PUQ.performance import performanceModel
import matplotlib.pyplot as plt
from PUQ.performanceutils.utils import plot_acc, plot_acqtime, plot_endtime, plot_errorend
import numpy as np

timeparams = [0.008, 0.006, 0.004]
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
acclevel = 0.1
result = []
n = 2048
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

labs = [r'$\mathcal{A}_1$', '$\mathcal{A}_2$', '$\mathcal{A}_3$']
fig, axes = plt.subplots(1, 1, figsize=(5, 5)) 
plot_acc(axes, n, acclevel, result, labellist=labs, logscale=False, fontsize=20)
axes.legend(bbox_to_anchor=(1, -0.2), ncol=len(labs), fontsize=15)
plt.show()

#####

scale_list = [1, 1.1, 1.2]
acclevel = 0.2
result = []
n = 2048
n0 = 128
batches = [1, 64, 128]
for bid, b in enumerate(batches):
    if b == 1:
        cons = 1
    else:
        cons             = np.arange(scale_list[bid], 1, -(scale_list[bid]-1)/n)[0:n] 
                        
    PM = performanceModel(worker=128, batch=b, n=n, n0=n0)
    PM.gen_gentime(0.1, typeGen='constant')
    PM.gen_simtime(1, 1, typeSim='normal', seed=1)
    PM.gen_accuracy(-1, 0.2, typeAcc='exponential')
    PM.acc = cons*PM.acc
    PM.simulate()
    PM.summarize()

    PM.complete(acclevel)
    result.append(PM)

labs = ['$b=1$', '$b=64$', '$b=128$']
fig, axes = plt.subplots(1, 1, figsize=(5, 5)) 
plot_acc(axes, n, acclevel, result, labellist=labs, logscale=False, fontsize=20, n0=n0)
axes.legend(bbox_to_anchor=(1, -0.2), ncol=len(labs), fontsize=15)
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(5, 5)) 
plot_acqtime(axes, n, acclevel, result, labellist=labs, logscale=False, fontsize=20, ind=True, n0=n0)
axes.legend(bbox_to_anchor=(1, -0.2), ncol=len(labs), fontsize=15)
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(5, 5)) 
plot_acqtime(axes, n, acclevel, result, labellist=labs, logscale=False, fontsize=20, ind=False, n0=n0)
axes.legend(bbox_to_anchor=(1, -0.2), ncol=len(labs), fontsize=15)
plt.show()

cons1             = np.arange(scale_list[bid], 1, -(scale_list[bid]-1)/n)[0:n] 
cons2             = np.arange(scale_list[bid], 1, -(scale_list[bid]-1)/n)[0:n] 
#####
scale_list = [1, 1.1, 1.2]
acclevel = 0.2
result = []
n = 2048
n0 = 128
batches = [1, 64]
for bid, b in enumerate(batches):
    if b == 1:
        cons = 1
    else:
        cons             = np.arange(scale_list[bid], 1, -(scale_list[bid]-1)/n)[0:n] 
                        
    PM = performanceModel(worker=128, batch=b, n=n, n0=n0)
    PM.gen_gentime(0.01, 0.1, typeGen='linear')
    PM.gen_simtime(1, 1, typeSim='normal', seed=1)
    PM.gen_accuracy(-1, 0.2, typeAcc='exponential')
    PM.acc = cons*PM.acc
    PM.simulate()
    PM.summarize()

    PM.complete(acclevel)
    result.append(PM)

labs = ['b=1', 'b=64', 'b=128']

fig, axes = plt.subplots(1, 1, figsize=(5, 5)) 
plot_acqtime(axes, n, acclevel, result, labellist=labs, logscale=False, fontsize=20, ind=True, n0=n0)
axes.legend(bbox_to_anchor=(1, -0.2), ncol=len(labs), fontsize=15)
plt.show()


scale_list = [1, 1.1, 1.2]
acclevel = 0.2
result = []
n = 2048
n0 = 128
batches = [1, 64]
for bid, b in enumerate(batches):
    if b == 1:
        cons = 1
    else:
        cons             = np.arange(scale_list[bid], 1, -(scale_list[bid]-1)/n)[0:n] 
                        
    PM = performanceModel(worker=128, batch=b, n=n, n0=n0)
    PM.gen_gentime(0.01, 0.1, 0.1, typeGen='quadratic')
    PM.gen_simtime(1, 1, typeSim='normal', seed=1)
    PM.gen_accuracy(-1, 0.2, typeAcc='exponential')
    PM.acc = cons*PM.acc
    PM.simulate()
    PM.summarize()

    PM.complete(acclevel)
    result.append(PM)

labs = ['b=1', 'b=64']

fig, axes = plt.subplots(1, 1, figsize=(5, 5)) 
plot_acqtime(axes, n, acclevel, result, labellist=labs, logscale=False, fontsize=20, ind=True, n0=n0)
axes.legend(bbox_to_anchor=(1, -0.2), ncol=len(labs), fontsize=15)
plt.show()