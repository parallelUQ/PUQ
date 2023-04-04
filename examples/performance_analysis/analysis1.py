from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import plot_accuracy, plot_acc, plot_acqtime, plot_endtime, plot_errorend
import matplotlib.pyplot as plt 


timeparams = [0.008, 0.006, 0.004]
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
acclevel = 0.1
result = []
n = 4096
worker = 1
for mid, m in enumerate(timeparams):
    PM = performanceModel(worker=1, batch=1, n=n)
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


# Batch version
n = 2048
timeparams = [1]
batches = [1, 8, 64]
accparams = [[-1, 0.2], [-1, 0.3], [-1, 0.4]]
result = []
worker = 64
for mid, m in enumerate(batches):
    PM = performanceModel(worker=worker, batch=batches[mid], n=n)
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

# Check
print(PM.complete_time)
print(PM.complete_no)
print(PM.acc_threshold)
print(PM.total_acq_time)
print(PM.completed_stage)

for job in PM.job_list:
    if job['endid'] == PM.complete_no:
        print(PM.complete_time)
        print(job['end'])
