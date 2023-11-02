
import numpy as np
import matplotlib.pyplot as plt
from ptest_funcs import sinfunc

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
        fvec[t_id, x_id] = cls_data.function(x, t)[0]
    plt.plot(x_vec, fvec[t_id, :], label=r'$\theta=\pi/$' + str(thlabel[t_id]), color=colors[t_id]) 

    
for d_id in range(len(cls_data.x)):
        plt.scatter(cls_data.x[d_id, 0], cls_data.real_data[0, d_id], color='black')
ft = 16
plt.xlabel(r'$x$', fontsize=ft)
plt.ylabel(r'$\eta(x, \theta)$', fontsize=16)
plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
plt.xticks(fontsize=ft-2)
plt.yticks(fontsize=ft-2)
plt.legend(bbox_to_anchor=(0.85, -0.2), fontsize=ft-2, ncol=2)
plt.savefig("Figure3a.png", bbox_inches="tight")
plt.show()

s = 1
cls_data = sinfunc()
dt = len(cls_data.true_theta)
cls_data.realdata(x=np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None], seed=s, isbias=True)

th_vec = [np.pi/5]
thlabel = [5]
x_vec  = (np.arange(0, 100, 1)/100)[:, None]
fvec   = np.zeros((len(th_vec), len(x_vec)))
colors = ['blue', 'orange', 'red', 'green', 'purple']
biasvec = cls_data.bias(x_vec)
for t_id, t in enumerate(th_vec):
    for x_id, x in enumerate(x_vec):
        fvec[t_id, x_id] = cls_data.function(x, t)[0]

    plt.plot(x_vec, fvec[t_id, :], label=r'$\eta(x, \theta=\pi/5)$', color='red') 
    plt.plot(x_vec, fvec[t_id, :] + biasvec.flatten(), label=r'$\mathbb{E}[y(x)]$', color='purple', linestyle='dashed') 
    
for d_id in range(len(cls_data.x)):
        plt.scatter(cls_data.x[d_id, 0], cls_data.real_data[0, d_id], color='black')
ft = 16
plt.xlabel(r'$x$', fontsize=ft)
plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
plt.xticks(fontsize=ft-2)
plt.yticks(fontsize=ft-2)
plt.legend(bbox_to_anchor=(0.85, -0.2), fontsize=ft-2, ncol=2)
plt.savefig("Figure3b.png", bbox_inches="tight")
plt.show()
