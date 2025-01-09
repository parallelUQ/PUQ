from PUQ.utils import parse_arguments, read_output
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from test_funcs import bimodal, banana, unimodal
from sir_funcs import SIR, SEIRDS
from utilities import test_data_gen, twoD
import scipy.stats as sps
    
def read_data(rep0=0, 
              repf=10, 
              methods=['ivar', 'imse', 'unif'], 
              batches=[8, 16, 32, 64], 
              examples=['unimodal', 'banana', 'bimodal'], 
              ids=['3', '1', '2'],
              ee='exploit',
              metric='TV', 
              folderpath=None, 
              ntotal=192):
    
    datalist1, datalist2 = [], []
    for eid, example in enumerate(examples):
        for m in methods:
            for bid, b in enumerate(batches):
                path = folderpath + ids[eid] + '_' + example + '_' + ee + '/'  + str(b) + '/'
                for r in range(rep0, repf):
                    desobj = read_output(path, example, m, b+1, b, r)
                    reps0 = desobj._info['reps0']
                    theta0 = desobj._info['theta0']
                    
                    theta = desobj._info['theta']
                    f = desobj._info['f']
                    
    return theta, f

theta, f = read_data(rep0=1, 
                     repf=2, 
                     methods=['ivar'], 
                     batches=[64], 
                     examples=['banana'], 
                     ids=['1'],
                     ee='exploit',
                     metric='TV', 
                     folderpath='/Users/ozgesurer/Desktop/fake/', 
                     ntotal=640)

thetach = theta[200:,]
fch = f[:, 200:]

print(np.unique(thetach, axis=0).shape)
print(np.unique(fch, axis=1).shape)

example = 'banana'
cls_func = eval(example)()
        
thetau = np.unique(thetach, axis=0)
a = []
for t in thetau:
    ar = np.all(thetach == t, axis=1).sum()
    a.append(ar)
    values = fch[:, np.all(thetach == t, axis=1)]
    meanval = cls_func.function(t[0], t[1])
    varval = cls_func.noise(t[None, :])
    
    if example == 'unimodal':
        plt.hist(values.flatten())
        plt.axvline(x=meanval, color='r', linestyle='--', linewidth=2)
        plt.axvline(x=np.mean(values.flatten()), color='y', linestyle='-', linewidth=2)
        plt.title('n:' + str(ar) + ' ' + 'm:' + str(np.round(meanval, 1)) + ' ' + 'v:' + str(np.round(varval, 2)) + ' ' + str(np.round(np.var(values.flatten()), 3)))
        plt.show()
    else:
        
        plt.hist(values[0, :].flatten())
        plt.axvline(x=meanval[0], color='r', linestyle='--', linewidth=2)
        plt.axvline(x=np.mean(values[0, :].flatten()), color='y', linestyle='-', linewidth=2)
        plt.title('n:' + str(ar) + ' ' + 'm:' + str(np.round(meanval[0], 1)) + ' ' + 'v:' + str(np.round(varval[0], 3)) + ' ' + str(np.round(np.var(values[0, :].flatten()), 3)))
        plt.show()
        
                
        plt.hist(values[1, :].flatten())
        plt.axvline(x=meanval[1], color='r', linestyle='--', linewidth=2)
        plt.axvline(x=np.mean(values[1, :].flatten()), color='y', linestyle='-', linewidth=2)
        plt.title('n:' + str(ar) + ' ' + 'm:' + str(np.round(meanval[1], 1)) + ' ' + 'v:' + str(np.round(varval[1], 3)) + ' ' + str(np.round(np.var(values[1, :].flatten()), 3)))
        plt.show()
        

print(np.sum(a))
    