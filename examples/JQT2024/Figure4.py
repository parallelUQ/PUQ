import numpy as np
import matplotlib.pyplot as plt
from ptest_funcs import pritam

# fig, axs = plt.subplots(1, 2, figsize=(15, 6))
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
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
a = np.arange(nmesh + 1) / nmesh
b = np.arange(nmesh + 1) / nmesh
X, Y = np.meshgrid(a, b)
Z = np.zeros((nmesh + 1, nmesh + 1))
B = np.zeros((nmesh + 1, nmesh + 1))
for i in range(nmesh + 1):
    for j in range(nmesh + 1):
        xt_cand = np.array([X[i, j], Y[i, j]]).reshape(1, 2)
        Z[i, j] = cls_data.function(X[i, j], Y[i, j], 0.5)
        B[i, j] = cls_data.bias(X[i, j], Y[i, j])
# fig, ax = plt.subplots()
CS = axs[0].contourf(X, Y, Z, cmap="Purples", alpha=0.75)
fig.colorbar(CS)
for xid1 in range(len(x)):
    for xid2 in range(len(y)):
        axs[0].scatter(
            x[xid1], x[xid2], marker="x", c="black", linewidth=3, s=100, zorder=2
        )
ft = 16
axs[0].set_xlim(-0.05, 1.05)
axs[0].set_ylim(-0.05, 1.05)
axs[0].set_xlabel(r"$x_1$", fontsize=ft)
axs[0].set_ylabel(r"$x_2$", fontsize=ft)
axs[0].set_xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft - 2)
axs[0].set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft - 2)
# plt.savefig("Figure4a.png", bbox_inches="tight")
# plt.show()

# Bias
# CS = plt.contour(X, Y, B, cmap='Purples', alpha=0.75)
# plt.clabel(CS, inline=1, fontsize=14)
# for xid1 in range(len(x)):
#     for xid2 in range(len(y)):
#         plt.scatter(x[xid1], x[xid2], marker='x', c='black', linewidth=3, s=100, zorder=2)

# plt.xlim(-0.02, 1.02)
# plt.ylim(-0.02, 1.02)
# plt.xlabel(r'$x_1$', fontsize=20)
# plt.ylabel(r'$x_2$', fontsize=20)
# plt.xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
# plt.yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
# plt.show()

# Model + Bias
# fig, ax = plt.subplots()
CS = axs[1].contourf(X, Y, Z + B, cmap="Purples", alpha=0.75)
fig.colorbar(CS)
for xid1 in range(len(x)):
    for xid2 in range(len(y)):
        axs[1].scatter(
            x[xid1], x[xid2], marker="x", c="black", linewidth=3, s=100, zorder=2
        )

axs[1].set_xlim(-0.05, 1.05)
axs[1].set_ylim(-0.05, 1.05)
axs[1].set_xlabel(r"$x_1$", fontsize=ft)
axs[1].set_ylabel(r"$x_2$", fontsize=ft)
axs[1].set_xticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft - 2)
axs[1].set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=ft - 2)
plt.savefig("Figure4.jpg", format="jpeg", bbox_inches="tight", dpi=1000)
plt.show()
