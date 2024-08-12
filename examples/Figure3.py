
import numpy as np
import matplotlib.pyplot as plt
from ptest_funcs import sinfunc

# optional for presentation 
#fig, axs = plt.subplots(1, 2, figsize=(24, 6))
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
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
    axs[0].plot(x_vec, fvec[t_id, :], label=r'$\theta=\pi/$' + str(thlabel[t_id]), color=colors[t_id], linewidth=3) 
    
for d_id in range(len(cls_data.x)):
        axs[0].scatter(cls_data.x[d_id, 0], cls_data.real_data[0, d_id], color='black', s=50)
# ft = 16         
ft = 16 
axs[0].set_xlabel(r'$x$', fontsize=ft)
axs[0].set_ylabel(r'$\eta(x, \theta)$', fontsize=ft)
axs[0].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
axs[0].tick_params(labelsize=ft-2)
axs[0].legend(bbox_to_anchor=(1.1, -0.2), fontsize=ft-2, ncol=4)

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

    axs[1].plot(x_vec, fvec[t_id, :], label=r'$\eta(x, \theta=\pi/5)$', color='red', linewidth=3) 
    axs[1].plot(x_vec, fvec[t_id, :] + biasvec.flatten(), label=r'$\mathbb{E}[y(x)]$', color='purple', linestyle='dashed', linewidth=3) 
    
for d_id in range(len(cls_data.x)):
        axs[1].scatter(cls_data.x[d_id, 0], cls_data.real_data[0, d_id], color='black', s=50)
ft = 16
axs[1].set_xlabel(r'$x$', fontsize=ft)
axs[1].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
axs[1].tick_params(labelsize=ft-2)
axs[1].legend(bbox_to_anchor=(0.9, -0.2), fontsize=ft-2, ncol=2)

plt.savefig('Figure3.jpg', format='jpeg', bbox_inches="tight", dpi=1000)
plt.show()
