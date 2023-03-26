from PUQ.performance import performanceModel
from PUQ.performanceutils.utils import plot_accuracy
import numpy as np
from result_read import get_rep_data   
import matplotlib.pyplot as plt 

n = 2048
w = 2
b = 1
rep = 30
example = 'himmelblau_ex'
s = 'himmelblau'
label = ['hybrid_ei_c1000/', 'hybrid_ei_c100/', 'hybrid_ei_c10/']
path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/'


result = []
for mid, m in enumerate(label):
    
    PM = performanceModel(worker=1, batch=1, n=n)
    
    ## ##
    filename = path + 'performanceAnalytics/new_fun_all/new_examples/' + example + '/' + label[mid]
    avgae, avgtime = get_rep_data(s, w, b, rep, filename, 'hybrid_ei')
    ## ##    
    
    ## ##
    xt    = np.arange(0, len(avgtime))
    xtest = np.arange(0, n)
    PM.gen_gentime(xt, avgtime, xtest, typeGen='regress')
    ## ##  
    
    ## ##
    PM.gen_simtime(0.0001, 0.0001, typeSim='normal')
    ## ##
    
    ## ##
    minl = np.min(avgae)
    maxl = np.max(avgae)
    lnew = [(litem - 0)/(maxl - 0) for litem in avgae]

    x_a  = np.log(np.arange(1, len(lnew)+1)) 
    y_a  = np.log(lnew)
    xtest_a  = np.log(np.arange(1, n+1))
    PM.gen_accuracy(x_a, y_a, xtest_a, typeAcc='regress')
    PM.acc = np.exp(PM.acc)
    ## ##
    
    PM.simulate()
    
    PM.summarize()
    result.append(PM)

plot_accuracy(result, n=n, acclevel=0.001, labellist=['M1', 'M2', 'M3'], logscale=True)



n = 2048
label = 'hybrid_ei_c1000/'
batches = [1, 64, 128, 256]
result = []
timeparams = [1.8, 1.6, 1.4, 1.2]
scale_list = [1, 1.25, 1.5, 1.75]
for mid, b in enumerate(batches):
    if b == 1:
        cons = 1
    else:
        cons             = np.arange(scale_list[mid], 1, -(scale_list[mid]-1)/n)[0:n] 
        
    PM = performanceModel(worker=512, batch=b, n=n)
    
    ## ##
    filename = path + 'performanceAnalytics/new_fun_all/new_examples/' + example + '/' + label
    avgae, avgtime = get_rep_data(s, w, 1, rep, filename, 'hybrid_ei')
    ## ##    
    
    ## ##
    PM.gen_gentime(timeparams[mid], typeGen='constant')
    ## ##  
    
    ## ##
    PM.gen_simtime(512, 0.1, typeSim='normal')
    ## ##
    
    ## ##
    minl = np.min(avgae)
    maxl = np.max(avgae)
    lnew = [(litem - 0)/(maxl - 0) for litem in avgae]

    x_a      = np.log(np.arange(1, len(lnew)+1)) 
    y_a      = np.log(lnew)
    xtest_a  = np.log(np.arange(1, n+1))
    PM.gen_accuracy(x_a, y_a, xtest_a, typeAcc='regress')
    PM.acc = cons*np.exp(PM.acc)
    ## ##
    
    PM.simulate()
    
    PM.summarize()
    result.append(PM)

plot_accuracy(result, n=n, acclevel=0.00001, labellist=['1', '64', '128', '256'], worker=512, logscale=True)