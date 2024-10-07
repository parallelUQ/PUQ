
from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
from ptest_funcs import pritam

def plot_des_pri(path, out, ex_name, w, batch, repid, methods, n0, nf):

    # fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for mid, method in enumerate(methods):
        design_saved = read_output(path + out + '/', ex_name, method, w, batch, repid)
        xt       = design_saved._info['theta']
        xacq = xt[n0:nf, 0:2]
        tacq = xt[n0:nf, 2]
        xinit = xt[0:n0, 0:2]
        ft = 16
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
                B[i, j] = cls_data.bias(X[i, j], Y[i, j])
        # plt.figure(figsize=(6, 5))
        axs[mid].contourf(X, Y, Z, cmap='Purples', alpha=0.75)
        
        for xid1 in range(len(x)):
            for xid2 in range(len(y)):
                axs[mid].scatter(x[xid1], x[xid2], marker='x', c='black', linewidth=3, s=100, zorder=2)
        
        unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    
        for val in unq:
            exist = False
            for xd in xr:
                if (val[0] == xd[0]) & (val[1] == xd[1]):
                    exist = True
                    axs[mid].scatter(val[0], val[1], marker='x', c='black', linewidth=3, s=100, zorder=2)
            if exist == False:
                axs[mid].scatter(val[0], val[1], marker='+', color='red', linewidth=3, s=100)
    
    
        axs[mid].scatter(xinit[:, 0], xinit[:, 1], marker='*', color='blue')
        for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
            axs[mid].annotate(label, xy=(x_count, y_count), xytext=(7, -7), textcoords='offset points', fontsize=ft-2)
    
        axs[mid].set_xlim(-0.05, 1.05)
        axs[mid].set_ylim(-0.05, 1.05)
        axs[mid].set_xlabel(r'$x_1$', fontsize=ft)
        axs[mid].set_ylabel(r'$x_2$', fontsize=ft)
        axs[mid].set_xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft-2)
        axs[mid].set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft-2)
    plt.savefig('Figure7.jpg', format='jpeg', bbox_inches="tight", dpi=1000)
    plt.show()
    

# choose either 'pritam' or 'sinfunc'
ex = 'pritam'
is_bias = False
n0, nf = 30, 60
outs = 'pritam'    
methods = ['ceivar', 'ceivarx']
batch = 1
worker = 2
repid = 1
#path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/newPUQ/examples/' 
#path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/denoise/' 
path = '/Users/ozgesurer/Desktop/JQT_experiments/'
plot_des_pri(path, outs, ex, worker, batch, repid, methods, n0, nf)