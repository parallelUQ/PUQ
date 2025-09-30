import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches
    
yellow_colors = [
    (1, 1, 1),
    (1, 1, 0.8),  # light yellow
    (1, 1, 0.6),
    (1, 1, 0.4),
    (1, 1, 0.2),
    (1, 1, 0),  # yellow
    (1, 0.8, 0),  # yellow-orange
    (1, 0.6, 0),  # orange
    (1, 0.4, 0),  # dark orange
    (1, 0.2, 0),  # very dark orange
    (1, 0, 0),  # very dark orange
]
yellow_cmap = ListedColormap(yellow_colors, name="yellow")
    
def heatmap(Xpl, Ypl, noise_grid, f_grid, p_grid, t_sample=None):

    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cs = ax[0].contourf(Xpl, Ypl, noise_grid.reshape(50, 50), cmap=yellow_cmap, alpha=0.75)
    cbar = fig.colorbar(cs)
    CS = ax[0].contour(Xpl, Ypl, f_grid.reshape(50, 50), colors="black")
    ax[0].clabel(CS, inline=True, fontsize=10)
    ax[0].set_xlabel(r"$\theta_1$", fontsize=16)
    ax[0].set_ylabel(r"$\theta_2$", fontsize=16)
    ax[0].tick_params(axis="both", labelsize=16)
    
    cs = ax[1].contourf(Xpl, Ypl, noise_grid.reshape(50, 50), cmap=yellow_cmap, alpha=0.75)
    cbar = fig.colorbar(cs)
    CS = ax[1].contour(Xpl, Ypl, p_grid.reshape(50, 50), cmap="coolwarm")
    if t_sample is not None:
        ax[1].scatter(t_sample[:, 0], t_sample[:, 1])
    ax[1].clabel(CS, inline=True, fontsize=10)
    ax[1].set_xlabel(r"$\theta_1$", fontsize=16)
    ax[1].set_ylabel(r"$\theta_2$", fontsize=16)
    ax[1].tick_params(axis="both", labelsize=16)
    plt.show()
    
def heatmap_pritam(cls_data):
    # test data
    nmesh = 50
    x1 = np.linspace(cls_data.zlim[0][0], cls_data.zlim[0][1], nmesh)
    x2 = np.linspace(cls_data.zlim[1][0], cls_data.zlim[1][1], nmesh)
    X1, X2 = np.meshgrid(x1, x2)
    Xg = np.vstack([X1.ravel(), X2.ravel()]).T
    #cls_data.theta_true[0]
    ng = np.array([cls_data.noise(x[0], x[1], cls_data.theta_true[0]) for x in Xg])
    fg = np.array([cls_data.function(x[0], x[1], cls_data.theta_true[0]) for x in Xg])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cs = ax[0].contourf(X1, X2, fg.reshape(50, 50), cmap=yellow_cmap, alpha=0.75)
    cbar = fig.colorbar(cs)
    #ax[1].clabel(CS, inline=True, fontsize=10)
    for x in cls_data.x:
        ax[0].scatter(x[0], x[1], marker="*", color="black")
    ax[0].set_xlabel(r"$x_1$", fontsize=16)
    ax[0].set_ylabel(r"$x_2$", fontsize=16)
    ax[0].tick_params(axis="both", labelsize=16)
    
    cs = ax[1].contourf(X1, X2, ng.reshape(50, 50), cmap=yellow_cmap, alpha=0.75)
    cbar = fig.colorbar(cs)
    #ax[0].clabel(CS, inline=True, fontsize=10)
    for x in cls_data.x:
        ax[1].scatter(x[0], x[1], marker="*", color="black")
    ax[1].set_xlabel(r"$x_1$", fontsize=16)
    ax[1].set_ylabel(r"$x_2$", fontsize=16)
    ax[1].tick_params(axis="both", labelsize=16)
    plt.show()
    






def toy_example(cls_data, Xpl, Ypl, fg, ng, persis_info):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    ft = 12
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=False)  # Disable constrained_layout
    fig.subplots_adjust(wspace=0.4, left=0.1, right=0.9)  # Adjust space between subplots
    
    # Contour plot for axs[1]
    divider = make_axes_locatable(axs[0])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)  # Adjust pad for the colorbar
    cs1 = axs[0].contourf(Xpl, Ypl, fg.reshape(50, 50), cmap=yellow_cmap, alpha=0.75)
    fig.colorbar(cs1, cax=cax1)
    
    xt_joint = np.column_stack((cls_data.x, np.repeat(cls_data.theta_true, len(cls_data.x))))
    axs[0].scatter(xt_joint[:, 0], xt_joint[:, 1], marker="x", c="black")
    axs[0].set_xlabel(r"$x$", fontsize=ft)
    axs[0].set_ylabel(r"$\vartheta$", fontsize=ft)
    axs[0].tick_params(axis="both", labelsize=ft)
    
    # Contour plot for axs[2]
    divider = make_axes_locatable(axs[1])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)  # Adjust pad for the colorbar
    cs2 = axs[1].contourf(Xpl, Ypl, ng.reshape(50, 50), cmap=yellow_cmap, alpha=0.75)
    fig.colorbar(cs2, cax=cax2)
    
    axs[1].scatter(xt_joint[:, 0], xt_joint[:, 1], marker="x", c="black")
    axs[1].set_xlabel(r"$x$", fontsize=ft)
    axs[1].set_ylabel(r"$\vartheta$", fontsize=ft)
    axs[1].tick_params(axis="both", labelsize=ft)
    
    # Scatter plot for axs[0]
    xs = np.linspace(cls_data.zlim[0][0], cls_data.zlim[0][1], 100)
    for k in range(10):
        axs[2].scatter(xs, 
                       np.array([cls_data.sim_f(np.array([x, cls_data.theta_true[0]]), persis_info) for x in xs]),
                       facecolors="none", edgecolors="green", s=80, linewidth=2)
    axs[2].plot(xs, 
                np.array([cls_data.function(x, cls_data.theta_true[0]) for x in xs]),
                linestyle="dotted", linewidth=3, color="black")
    axs[2].set_xlabel(r"$x$", fontsize=ft)
    #axs[2].set_ylabel(r"$\theta$", fontsize=ft)
    axs[2].tick_params(axis="both", labelsize=ft)
    
    plt.show()



def visual_xt(cls_data, des_obj, z0u, rep0, tg, pg, ng, Xpl, Ypl, new=True):
    
    z, reps = des_obj.xt, des_obj.reps
    
    ft = 14
    fig, ax = plt.subplots(figsize=(5, 4))

    for xitem in cls_data.x:
        ax.vlines(
            xitem, 0, 1, linestyles="dotted", colors="green", linewidth=3, zorder=5
        )  
    for label, x_count, y_count in zip(reps, z[:, 0], z[:, 1]):
        if new:
            if np.any(np.all(z0u == np.array([x_count, y_count]), axis=1)):
                col = "cyan"
            else:
                col = "blue"
        else:
            if label > rep0:
                col = "blue"
            else:
                col = "cyan"                

        plt.annotate(
            label,
            xy=(x_count, y_count),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=ft-2,
            color=col,
            weight="bold",
            zorder=6,
        )
    
    ax.contourf(Xpl, Ypl, ng.reshape(50, 50), cmap=yellow_cmap, alpha=0.75)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x$", fontsize=ft)
    ax.set_ylabel(r"$\theta$", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    ax2 = ax.twiny()
    ax2.plot(pg, tg, color='black', label="Y-axis scatter")
    ax2.set_xlabel(r"$p(y|\theta)$", fontsize=ft)
    ax2.tick_params(axis="both", labelsize=ft)
    plt.show()


def fig6(desobj, theta0, rep0, Xpl, Ypl, pg, ng, ax, figset={}, fig={}, exp=True):
    
    th, reps = desobj.xt[:, 1:3], desobj.reps
    
    xtick, ytick, cbarf = figset.get("xtick", True), figset.get("ytick", True), figset.get("cbar", True)

    ft = 12
    nm = Xpl.shape[0]
    cs = ax.contourf(Xpl, Ypl, ng.reshape(nm, nm), cmap=yellow_cmap, alpha=0.75)  
    cp = ax.contour(Xpl, Ypl, pg.reshape(nm, nm), cmap="coolwarm")   
    
    
    if cbarf:
        cbar = fig.colorbar(cs, ax=ax, pad=0.1)
        
    for label, x_count, y_count in zip(reps, th[:, 0], th[:, 1]):
        
        if exp:
            if np.any(np.all(theta0 == np.array([x_count, y_count]), axis=1)):
                col = "cyan"
            else:
                col = "blue"  
        else:
            if label > rep0:
                col = "blue"  
            else:
                col = "cyan"

        ax.annotate(
            label,
            xy=(x_count, y_count),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=ft,
            color=col
        )
    
    if xtick:
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(r"$\vartheta_1$", fontsize=ft)
    else:
        ax.set_xticks([])
        
    if ytick:
        ax.set_yticks([0, 0.5, 1]) 
        ax.set_ylabel(r"$\vartheta_2$", fontsize=ft)
    else:
        ax.set_yticks([])   
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=ft)



def visual_pritam(des_obj, z0u):
    ft = 16
    xtu, repu = des_obj.xt, des_obj.reps
    for xtid, xt in enumerate(xtu):
        # print(xt)
        if xt in z0u:
            col = "cyan"
        else:
            col = "blue"
        
        plt.annotate(
            repu[xtid],
            xy=(xt[0], xt[1]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=ft,
            color=col,
            weight="bold",
        )
        
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.show()
    
    ut = []
    for xtid, xt in enumerate(xtu):
        if xt not in z0u:
            ut.append(xt[2])
    plt.hist(ut)
    plt.show()
            
import os
import dill as pickle
def save_output(desing_obj, name, al_func, seedno, path=None, label=None):
    
    if path is None:
        if not os.path.isdir('output'):
            os.mkdir('output')
            
        design_path = 'output/' + name + '_' + al_func + '_seed_' + str(seedno) + "_" + label + '.pkl'
    else:
        design_path = path + name + '_' + al_func + '_seed_' + str(seedno) + "_" + label + '.pkl'
        
    with open(design_path, 'wb') as file:
        pickle.dump(desing_obj, file)
    
def read_output(path1, name, al_func, seedno, label=None):
    
    design_path = path1 + 'output/' + name + '_' + al_func + '_seed_' + str(seedno) + "_" + label + '.pkl'
    with open(design_path, 'rb') as file:
        design_obj = pickle.load(file) 

    return design_obj



def create_entry(desobj, fname, method, s, h):
    return [
        {"MSE": he["MSE"], 
         "MAD": he["MAD"], 
         "VAR": he["VAR"], 
         "MSEy": he["MSEy"], 
         "MADy": he["MADy"], 
         "MSEn": he["MSEn"], 
         "MADn": he["MADn"], 
         "t": he["t"], 
         "new": he["new"], 
         "method": method, 
         "example": fname, 
         "s": s, 
         "h": h, 
         "horizon": he["h"]}
        for he in desobj["H"]
    ]

def read_data(rep0=0, 
              repf=10, 
              methods=["ivar", "imse", "unif"],
              examples=["unimodal", "banana", "bimodal"],
              folderpath=None,
              label=None,
              ):

    datalist = []
    for eid, example in enumerate(examples):
        for mid, m in enumerate(methods):
            for r in range(rep0, repf):

                desobj = read_output(folderpath, example, m, r, label=label[mid])
                entry = create_entry(desobj, example, m, r, label[mid])
                datalist.extend(entry)

    df = pd.DataFrame(datalist)
    return df

def read_summary_metric(folderpath, examples, methods, ls, rep=(1,31)):
    dfl = []
    for eid, example in enumerate(examples):
        for mid, m in enumerate(methods):
            for r in range(rep[0], rep[1]):
    
                desobj = read_output(folderpath, example, m, r, label=ls[mid])
                for heid, he in enumerate(desobj.H):
                    sm = he["summary_metric"]
                    sm["r"] = r
                    sm["method"] = m
                    sm["h"] = ls[mid]
                    sm["t"] = heid
                    dfl.append(sm)
              
    df = pd.DataFrame(dfl)
    return df
