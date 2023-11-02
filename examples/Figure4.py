
import numpy as np
import matplotlib.pyplot as plt
from ptest_funcs import pritam


s = 1
x = np.linspace(0, 1, 3)
y = np.linspace(0, 1, 3)
xr = np.array([[xx, yy] for xx in x for yy in y])
xr = np.concatenate((xr, xr))

# Data
cls_data = pritam()
cls_data.realdata(xr, seed=s)

# Model
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
plt.savefig("Figure4a.png", bbox_inches="tight")
plt.show()

# Bias
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

# Model + Bias

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
plt.savefig("Figure4b.png", bbox_inches="tight")
plt.show()


#biastrue = cls_data.bias(cls_data.x[:, 0], cls_data.x[:, 1])
#biascheck = np.zeros(len(cls_data.x))

#for xid in range(len(cls_data.x)):
#    biascheck[xid] = cls_data.bias(cls_data.x[xid, 0], cls_data.x[xid, 1])
    
#print(biastrue - biascheck)