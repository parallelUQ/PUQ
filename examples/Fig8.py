
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
from ptest_funcs import pritam

def plot_des_pri(path, out, ex_name, w, b, repid, method, n0, nf):

    design_saved = read_output(path + out + '/', ex_name, method, w, b, repid)

    xt       = design_saved._info['theta']

    xacq = xt[n0:nf, 0:2]
    tacq = xt[n0:nf, 2]
    
    xinit = xt[0:n0, 0:2]
    
    #plt.hist(tacq)
    #plt.axvline(x =0.5, color = 'r')
    #plt.xlabel(r'$\theta$')
    #plt.xlim(0, 1)
    #plt.show()

    ft = 15

    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)
    xr = np.array([[xx, yy] for xx in x for yy in y])
    xr = np.concatenate((xr, xr))
    cls_data = pritam()
    cls_data.realdata(xr, seed=repid)
    nmesh = 50
    a = np.arange(nmesh+1)/nmesh
    b = np.arange(nmesh+1)/nmesh
    X, Y = np.meshgrid(a, b)
    Z = np.zeros((nmesh+1, nmesh+1))
    B = np.zeros((nmesh+1, nmesh+1))
    for i in range(nmesh+1):
        for j in range(nmesh+1):
            xt_cand = np.array([X[i, j], Y[i, j]]).reshape(1, 2)
            Z[i, j] = cls_data.function(X[i, j], Y[i, j], 0.5)
            B[i, j] =  cls_data.bias(X[i, j], Y[i, j])
    plt.figure(figsize=(5, 5))
    plt.contourf(X, Y, Z, cmap='Purples', alpha=0.5)
    
    for xid1 in range(len(x)):
        for xid2 in range(len(y)):
            plt.scatter(x[xid1], x[xid2], marker='x', c='black', s=60, zorder=2)
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)

    for val in unq:
        exist = False
        for xd in xr:
            if (val[0] == xd[0]) & (val[1] == xd[1]):
                exist = True
                plt.scatter(val[0], val[1], marker='x', c='black', s=60, zorder=2)
        if exist == False:
            plt.scatter(val[0], val[1], marker='+', color='red')

    
    
    
    plt.scatter(xinit[:, 0], xinit[:, 1], marker='*', color='blue')
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(7, -7), textcoords='offset points')

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'$x_1$', fontsize=ft)
    plt.ylabel(r'$x_2$', fontsize=ft)
    plt.xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft-5)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft-5)
    
    plt.show()



clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 

# choose either 'pritam' or 'sinfunc'
ex = 'pritam'
is_bias = False
if ex == 'pritam':
    n0, nf = 30, 60
    if is_bias:
        outs = 'pritam_bias'
        method = ['ceivarxbias', 'ceivarbias', 'lhs', 'rnd']
    else:
        outs = 'pritam'     
        # method = ['ceivarx', 'ceivar', 'lhs', 'rnd']
        method = ['ceivarx', 'ceivar']
        

        

labelsb = [r'$\mathcal{A}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$']


batch = 1
worker = 2
for repid in range(1, 2):
    #repid = 1
    fonts = 18
    
    path = '/Users/ozgesurer/Desktop/des_examples/newPUQ/examples/'
    
    
    for mid, m in enumerate(method):
        plot_des_pri(path, outs, ex, worker, batch, repid, m, n0=n0, nf=nf)
