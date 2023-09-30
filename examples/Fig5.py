
import numpy as np
import matplotlib.pyplot as plt
from ptest_funcs import sinfunc, pritam

s = 1
cls_data = sinfunc()
dt = len(cls_data.true_theta)
cls_data.realdata(x=np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None], seed=s)

th_vec = [np.pi/7, np.pi/6, np.pi/5, np.pi/4]
thlabel = [7, 6, 5, 4]
x_vec  = (np.arange(0, 100, 1)/100)[:, None]
fvec   = np.zeros((len(th_vec), len(x_vec)))
colors = ['blue', 'orange', 'red', 'green', 'purple']
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_data.function(x, t)
    plt.plot(x_vec, fvec[t_id, :], label=r'$\theta=\pi/$' + str(thlabel[t_id]), color=colors[t_id]) 

    
for d_id in range(len(cls_data.x)):
        plt.scatter(cls_data.x[d_id, 0], cls_data.real_data[0, d_id], color='black')
ft = 16
plt.xlabel(r'$x$', fontsize=ft)
plt.ylabel(r'$\eta(x, \theta)$', fontsize=16)
plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
plt.xticks(fontsize=ft-2)
plt.yticks(fontsize=ft-2)
plt.legend()
plt.show()



x = np.linspace(0, 1, 3)
y = np.linspace(0, 1, 3)
xr = np.array([[xx, yy] for xx in x for yy in y])
xr = np.concatenate((xr, xr))
cls_data = pritam()
cls_data.realdata(xr, seed=s)
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
plt.contourf(X, Y, Z, cmap='Purples', alpha=0.5)

for xid1 in range(len(x)):
    for xid2 in range(len(y)):
        plt.scatter(x[xid1], x[xid2], marker='x', c='black', s=60, zorder=2)

plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.xlabel(r'$x_1$', fontsize=20)
plt.ylabel(r'$x_2$', fontsize=20)
plt.xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
plt.yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
plt.show()


plt.contourf(X, Y, B, cmap='Purples', alpha=0.5)

for xid1 in range(len(x)):
    for xid2 in range(len(y)):
        plt.scatter(x[xid1], x[xid2], marker='x', c='black', s=60, zorder=2)

plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.xlabel(r'$x_1$', fontsize=20)
plt.ylabel(r'$x_2$', fontsize=20)
plt.xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
plt.yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
plt.show()

plt.contourf(X, Y, Z+B, cmap='Purples', alpha=0.5)

for xid1 in range(len(x)):
    for xid2 in range(len(y)):
        plt.scatter(x[xid1], x[xid2], marker='x', c='black', s=60, zorder=2)

plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.xlabel(r'$x_1$', fontsize=20)
plt.ylabel(r'$x_2$', fontsize=20)
plt.xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
plt.yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
plt.show()


biastrue = cls_data.bias(cls_data.x[:, 0], cls_data.x[:, 1])
biascheck = np.zeros(len(cls_data.x))

for xid in range(len(cls_data.x)):
    biascheck[xid] = cls_data.bias(cls_data.x[xid, 0], cls_data.x[xid, 1])
    
print(biastrue - biascheck)