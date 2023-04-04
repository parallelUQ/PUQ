from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import plot_accuracy, plot_accuracy2, plot_workers, plot_acc, plot_acqtime, plot_endtime, plot_errorend
import numpy as np
from result_read import get_rep_data   
import matplotlib.pyplot as plt 

n = 512
w = 2
b = 1
rep = 30
example = 'himmelblau_ex'
s = 'himmelblau'

label = 'hybrid_ei_c1000/'
path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/'

filename = path + 'performanceAnalytics/new_fun_all/new_examples/' + example + '/' + label
avgae, avgtime = get_rep_data(s, w, 1, rep, filename, 'hybrid_ei')
        
## ##
minl = np.min(avgae)
maxl = np.max(avgae)
lnew = [(litem - 0)/(maxl - 0) for litem in avgae]

x_a      = np.log(np.arange(1, len(lnew)+1)) 
y_a      = np.log(lnew)
xtest_a  = np.log(np.arange(1, n+1))

PM = performanceModel(worker=64, batch=1, n=n)
PM.gen_accuracy(x_a, y_a, xtest_a, typeAcc='regress')
fitted_accuracy = np.exp(PM.acc)
plt.plot(fitted_accuracy)
plt.yscale('log')
plt.xscale('log')
plt.show()
## ##

worker = 256    
acclevel = 0.000075
show = False
lab = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16', 'b=32', 'b=64', 'b=128', 'b=256']
if show:
    batches = [1, 4, 16, 64]
    timeparams = [1]
    scale_list = [1, 1.33, 1.67, 2]
    for sim in [10, 50, 100]:
        result = []
        for mid, b in enumerate(batches):
            if b == 1:
                cons = 1
            else:
                cons             = np.arange(scale_list[mid], 1, -(scale_list[mid]-1)/n)[0:n] 
                
            PM = performanceModel(worker=worker, batch=b, n=n)
            PM.gen_gentime(timeparams[0], typeGen='constant')
            PM.gen_simtime(sim, 0.1*sim, typeSim='normal')
            PM.gen_accuracy(cons*fitted_accuracy, typeAcc='batched')
            PM.simulate()
            PM.summarize()
            PM.complete(acclevel)
            result.append(PM)
        
        #plot_accuracy(result, n=n, acclevel=acclevel, labellist=['1', '4', '16', '64'], worker=64, logscale=True)

        fig, axes = plt.subplots(2, 2, figsize=(22, 20)) 
        plot_acc(axes[0, 0], n, acclevel, result, labellist=lab, logscale=True, fontsize=25)
        plot_acqtime(axes[0, 1], n, acclevel, result, labellist=lab, logscale=True, fontsize=25)
        plot_endtime(axes[1, 0], n, acclevel, result, labellist=lab, worker=worker, logscale=True, fontsize=25)
        plot_errorend(axes[1, 1], n, acclevel, result, labellist=lab, worker=worker, logscale=True, fontsize=25)
        
show = True
if show:

    n = 512
    label = 'hybrid_ei_c1000/'
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    timeparams = [1]
    scale_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] #[1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6]
    scale_list = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6]
    for sim in [0.1, 1, 10]:
        result = []
        for mid, b in enumerate(batches):
            if b == 1:
                cons = 1
            else:
                cons             = np.arange(scale_list[mid], 1, -(scale_list[mid]-1)/n)[0:n] 
                
            PM = performanceModel(worker=worker, batch=b, n=n)
            PM.gen_gentime(0.1, 0.1, typeGen='linear')
            PM.gen_simtime(sim, sim*0.1, typeSim='normal')
            PM.gen_accuracy(cons*fitted_accuracy, typeAcc='batched')
            PM.simulate()
            PM.summarize()
            PM.complete(acclevel)
            #plot_workers(PM, PM.job_list, PM.stage_list)
            result.append(PM)
        
        #plot_accuracy(result, n=n, acclevel=acclevel, labellist=['1', '4', '16', '64'], worker=64, logscale=True)   
        fig, axes = plt.subplots(2, 2, figsize=(22, 20)) 
        plot_acc(axes[0, 0], n, acclevel, result, labellist=lab, logscale=True, fontsize=25)
        plot_acqtime(axes[0, 1], n, acclevel, result, labellist=lab, logscale=True, fontsize=25)
        plot_endtime(axes[1, 0], n, acclevel, result, labellist=lab, worker=worker, logscale=False, fontsize=25)
        plot_errorend(axes[1, 1], n, acclevel, result, labellist=lab, worker=worker, logscale=False, fontsize=25)
