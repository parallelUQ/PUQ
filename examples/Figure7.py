
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
            B[i, j] =  cls_data.bias(X[i, j], Y[i, j])
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, cmap='Purples', alpha=0.75)
    
    for xid1 in range(len(x)):
        for xid2 in range(len(y)):
            plt.scatter(x[xid1], x[xid2], marker='x', c='black', linewidth=3, s=100, zorder=2)
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)

    for val in unq:
        exist = False
        for xd in xr:
            if (val[0] == xd[0]) & (val[1] == xd[1]):
                exist = True
                plt.scatter(val[0], val[1], marker='x', c='black', linewidth=3, s=100, zorder=2)
        if exist == False:
            plt.scatter(val[0], val[1], marker='+', color='red', linewidth=3, s=100)


    plt.scatter(xinit[:, 0], xinit[:, 1], marker='*', color='blue')
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(7, -7), textcoords='offset points', fontsize=ft-2)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'$x_1$', fontsize=ft)
    plt.ylabel(r'$x_2$', fontsize=ft)
    plt.xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft-2)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft-2)
    plt.savefig("Figure7_" + method + ".png", bbox_inches="tight")
    plt.show()
    
def FIG7(method, path, n0, nf, repid, batch, worker, outs):
    
    for mid, m in enumerate(method):
        plot_des_pri(path, outs, ex, worker, batch, repid, m, n0=n0, nf=nf)

# choose either 'pritam' or 'sinfunc'
ex = 'pritam'
is_bias = False
n0, nf = 30, 60
outs = 'pritam'    
method = ['ceivarx', 'ceivar']
batch = 1
worker = 2
repid = 1
#path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/newPUQ/examples/' 
path = '/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/denoise/' 
FIG7(method, path, n0, nf, repid, batch, worker, outs)