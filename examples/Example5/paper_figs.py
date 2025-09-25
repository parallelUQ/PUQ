#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:45:08 2025

@author: surero
"""
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_data, read_output, read_summary_metric
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap

# Custom yellow-orange colormap
yellow_colors = [
    (1, 1, 1),
    (1, 1, 0.8),
    (1, 1, 0.6),
    (1, 1, 0.4),
    (1, 1, 0.2),
    (1, 1, 0),
    (1, 0.9, 0),
    (1, 0.8, 0),
    (1, 0.6, 0),
    (1, 0.4, 0),
    (1, 0.2, 0),
]
yellow_cmap = ListedColormap(yellow_colors, name="yellow")


custom_palette = {"ivar": "red", "imse": "blue", "var": "green", "varx": "magenta"}
hue_order = ["ivar", "imse", "var"]

hatch_color_map = {
    (0.875, 0.125, 0.125, 1): "//",
    (0.125, 0.125, 0.875, 1): "\\",
    (0.06274509803921569, 0.4392156862745098, 0.06274509803921569, 1): "xx",
}




def lineplot(
    df,
    examples,
    metric="TV",
    ci=None,
    label=None,
    custom_labels=None,
    ax=None,
    hue=None,
    figset={},
    scaled=True,
):
    ifLeg = figset.get("ifLeg", True)
    sety = figset.get("sety", True)
    ft = figset.get("ft", 12)
    lw = figset.get("lw", 3)

    for i, example in enumerate(examples):
        df1 = df.loc[df["example"] == example]

        if scaled:
            group_cols = ["t", hue, "h"]
            mean_df = df1.groupby(group_cols)[metric].mean().reset_index()
            max_val = mean_df[metric].max()
            mean_df[metric] = mean_df[metric] / max_val
            data = mean_df.copy()
        else:
            data = df1.copy()

        sns.lineplot(
            data=data,
            x="t",
            y=metric,
            hue=hue,
            style="h",
            palette=custom_palette,
            errorbar=ci,
            linewidth=lw,
            ax=ax,
        )

        lgd = ax.legend(
            loc="upper center",
            bbox_to_anchor=(1.3, 1),
            fancybox=True,
            shadow=True,
            ncol=1,
            fontsize=ft,
        )

        ax.set_yscale("log")
        ax.set_xlabel("t", fontsize=ft)
        if sety == False:
            ax.set_ylabel("")
        else:
            ax.set_ylabel(metric, fontsize=ft)
        ax.tick_params(axis="both", labelsize=ft)
        if ifLeg == False:
            ax.legend().remove()


def boxplot(df, ax, x, hue, figset={}):
    ifLeg, sety = figset.get("ifLeg", True), figset.get("sety", True)
    ft = figset.get("ft", 12)

    sns.boxplot(
        x=x,
        y="percent",
        hue="method",
        data=df,
        ax=ax,
        showfliers=False,
        palette=custom_palette,
        hue_order=hue_order,
    )

    # select the correct patches
    patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]

    # iterate through the patches for each subplot
    for patch in patches:
        fc = patch.get_facecolor()
        patch.set_edgecolor(fc)
        patch.set_facecolor("none")
        patch.set_hatch(hatch_color_map[fc])

    # # # Fix legend to match hatches and colors
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        title="method",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    for patch in legend.get_patches():
        fc = patch.get_facecolor()
        patch.set_edgecolor(fc)
        patch.set_facecolor("none")
        patch.set_hatch(hatch_color_map[fc])

    if sety == False:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Percent of exploration", fontsize=ft)

    plt.xticks(fontsize=ft)

    if ifLeg == False:
        ax.legend().remove()


def plot_experiment_figure(
    examples,
    path,
    ms,
    ls,
    lg,
    ly,
    reps=(1, 31),
    metric="MAD",
    layout="2xN",
    figsize=(12, 6),
    ft=10,
    r0=5,
    n0=150,
    title=None,
):

    single_example = len(examples) == 1
    if layout == "1x2" and single_example:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
    elif layout == "2xN":
        fig, ax = plt.subplots(2, len(examples), figsize=figsize)
    else:
        raise ValueError("Unsupported layout or mismatched number of examples.")

    for i, ex in enumerate(examples):
        print(ex)
        figset = {"ifLeg": lg[i], "sety": ly[i], "ft": ft, "r0": r0, "n0": n0}
        df1 = read_data(
            rep0=reps[0],
            repf=reps[1],
            methods=ms,
            examples=[ex],
            folderpath=path,
            label=ls,
        )
        print(df1["s"])
        df2 = (
            df1.groupby(["s", "h", "example", "method"])["new"]
            .mean()
            .mul(100)
            .reset_index()
        )
        df2.rename(columns={"new": "percent"}, inplace=True)

        if layout == "1x2":
            lineplot(df1, [ex], metric=metric, hue="method", ax=ax[0], figset=figset)
            boxplot_alternative(path, ms, ls, ex, ax=ax[1], figset=figset)
            #boxplot(df=df2, ax=ax[1], x="h", hue="method", figset=figset)
        else:  # layout == "2xN"
            lineplot(df1, [ex], metric=metric, hue="method", ax=ax[0, i], figset=figset)
            boxplot_alternative(path, ms, ls, ex, ax=ax[1, i], figset=figset)
            #boxplot(df=df2, ax=ax[1, i], x="h", hue="method", figset=figset)

    plt.tight_layout(pad=0.2)
    if title is not None:
        plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.show()




def plot_SIR_experiment_figure(
    path, ms, ls, reps=(1, 31), metric="MAD", figsize=(12, 6), ft=10, figset={}, title=None,
):
    usual_boxplot = figset.get("usual_boxplot", True)
    shift, scaled = figset.get("shift_x", 0), figset.get("scaled", True)

    fig, ax = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"wspace": 0.25}
    )
    #, "width_ratios": [1.25, 1.25, 1, 1]
    lw = 3

    df1 = read_data(rep0=reps[0], repf=reps[1], methods=ms, examples=["SIR"], folderpath=path, label=ls)
    print(df1["s"])
    df3 = df1.loc[df1["t"] == 199]
    
 
    if scaled:
        group_cols = ["t", "method", "h"]
        mean_df = df1.groupby(group_cols)[metric].mean().reset_index()
        max_val = mean_df[metric].max()
        mean_df[metric] = mean_df[metric] / max_val
        data = mean_df.copy()
    else:
        data = df1.copy()

    # Lineplot
    sns.lineplot(
        data=data,
        x="t",
        y=metric,
        hue="method",
        style="h",
        palette=custom_palette,
        errorbar=None,
        linewidth=lw,
        ax=ax[0],
    )
    ax[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
        shadow=True,
        ncol=2,
        fontsize=ft,
    )
    ax[0].set_yscale("log")

    if usual_boxplot:
        df2 = (
            df1.groupby(["s", "h", "example", "method"])["new"]
            .mean()
            .mul(100)
            .reset_index()
        )
        df2.rename(columns={"new": "percent"}, inplace=True)
        box_data = [
            (df2, "percent", ax[1], "percent"),
            (df3, "MADy", ax[2], r"$MAD^y$"),
            (df3, "MADn", ax[3], r"$MAD^n$"),
        ]
    else:
        figset["legend_right"] = False
        boxplot_alternative(path, ms, ls, "SIR", ax=ax[1], figset=figset)#{"legend_right":False})
        box_data = [
            (df3, "MADy", ax[2], r"$MAD^y$"),
            (df3, "MADn", ax[3], r"$MAD^n$"),
        ]
    
    for boxid, box in enumerate(box_data):
        sns.boxplot(
            x=box[1],
            y="h",
            hue="method",
            data=box[0],
            showfliers=False,
            palette=custom_palette,
            hue_order=hue_order,
            ax=box[2],
        )

        # select the correct patches
        patches = [
            patch for patch in box[2].patches if type(patch) == mpl.patches.PathPatch
        ]

        # iterate through the patches for each subplot
        for patch in patches:
            fc = patch.get_facecolor()
            patch.set_edgecolor(fc)
            patch.set_facecolor("none")
            patch.set_hatch(hatch_color_map[fc])

        box[2].legend_.remove()

        # # # Fix legend to match hatches and colors
        if boxid == 1:
            handles, labels = box[2].get_legend_handles_labels()
            legend = box[2].legend(
                handles,
                labels,
                title="method",
                loc="lower center",
                bbox_to_anchor=(0.5+shift, -0.5),
                ncol=3,
            )
            box[2].set_yticks([])

            for patch in legend.get_patches():
                fc = patch.get_facecolor()
                patch.set_edgecolor(fc)
                patch.set_facecolor("none")
                patch.set_hatch(hatch_color_map[fc])

        for label in box[2].get_yticklabels():
            label.set_rotation(90)

        box[2].set_ylabel("")
        box[2].set_xlabel(box[3], fontsize=ft)
            
    if title is not None:
        plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.show()




def plot_summary_metric(path, examples, methods, ls, metrics, rep=(1,31)):

    df = read_summary_metric(path, examples, methods, ls, rep)

    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        sns.lineplot(
            data=df,
            x="t",
            y=metric,
            hue="method",
            style="h",
            ax=ax,
        )
        ax.set_yscale("log")
        plt.show()



def plot_sir_figure(path, label="adapt", methods=["ivar", "imse", "var"], ft=12, repid=(1, 2), title=None):

    # Prepare data
    from utils_SIR import test_gen_SIR
    from SIR_funcs import SIRx

    cls_func = SIRx()
    cls_func.realdata(x=np.array([[0.25, 0.25], [0.75, 0.75]]), seed=None)
    p_grid, t_grid, noise_grid, Xpl, Ypl = test_gen_SIR(cls_func, return_XY=False)
    
    
    for r in np.arange(repid[0], repid[1]):
        # Plotting
        fig, ax = plt.subplots(1, len(methods), figsize=(11, 3), constrained_layout=True)
    
        for mid, method in enumerate(methods):
            desobj = read_output(path, "SIR", method, r, label=label)
    
            # Contour plots
            ax[mid].contour(Xpl, Ypl, p_grid.reshape(50, 50), cmap="coolwarm", zorder=2)
            cs = ax[mid].contourf(
                Xpl,
                Ypl,
                np.sum(noise_grid, axis=1).reshape(50, 50),
                cmap=yellow_cmap,
                alpha=0.75,
                zorder=1,
            )
    
            # Add colorbar only to the last axis
            if mid == len(methods) - 1:
                fig.colorbar(cs, ax=ax[mid], pad=0.1)
    
            # Annotate sample locations and frequency
            unique_rows, counts = np.unique(
                desobj.zs[200:, 2:4], axis=0, return_counts=True
            )
            print(counts)
            for lbl, x, y in zip(counts, unique_rows[:, 0], unique_rows[:, 1]):
                ax[mid].annotate(
                    lbl,
                    xy=(x, y),
                    xytext=(0, 0),
                    textcoords="offset points",
                    fontsize=ft,
                    color="blue",
                    zorder=3,
                )
    
            # Axis formatting
            ax[mid].set_xticks([0, 0.5, 1])
            ax[mid].set_xlabel(r"$\theta_1$", fontsize=ft + 2)
            ax[mid].tick_params(axis="both", labelsize=ft + 2)
            if mid == 0:
                ax[mid].set_yticks([0, 0.5, 1])
                ax[mid].set_ylabel(r"$\theta_2$", fontsize=ft + 2)
            else:
                ax[mid].set_yticks([])
                
        if title is not None:
            plt.savefig(title, dpi=300, bbox_inches="tight")
        plt.show()

def plot_pritam(path, repid=1, label="target", title=None):
    from test_functions import pritam
    from utils_sample import test_data_gen_pri
    
    # Plot settings
    ft = 12
    ms = ["ivar", "imse", "var"]
    n0 = 150

    
    # Define test points
    x = np.array([0.2, 0.8])
    y = np.array([0.2, 0.8])
    xr = np.array([[xx, yy] for xx in x for yy in y])
    
    # Instantiate test function object and generate real data
    cex = pritam()
    cex.realdata(x=xr, seed=None)
    
    # Generate test grid
    nmesh = 50
    x1 = np.linspace(cex.zlim[0][0], cex.zlim[0][1], nmesh)
    x2 = np.linspace(cex.zlim[1][0], cex.zlim[1][1], nmesh)
    X1, X2 = np.meshgrid(x1, x2)
    Xg = np.vstack([X1.ravel(), X2.ravel()]).T
    
    # Evaluate noise and function values on the grid
    ng = np.array([cex.noise(x[0], x[1], cex.theta_true[0]) for x in Xg])
    fg = np.array([cex.function(x[0], x[1], cex.theta_true[0]) for x in Xg])
    
    # Create subplots
    fig, ax = plt.subplots(2, 3, figsize=(9, 5), constrained_layout=True)
    
    for mid, m in enumerate(ms):
        desobj = read_output(path, "pritam", m, repid, label="target")
    
        # Top row: histogram of theta samples
        ax[0, mid].hist(
            desobj.zs[n0:, 2],
            bins=30,
            edgecolor="black",
            alpha=0.75,
            color="blue"
        )
        ax[0, mid].set_xlim(0, 1)
        ax[0, mid].set_xlabel(r"$\vartheta$", fontsize=ft)
        if mid == 0:
            ax[0, mid].set_ylabel("Frequency", fontsize=ft)
    
        # Bottom row: scatter plot of x-samples with contour background
        ax[1, mid].scatter(
            desobj.zs[n0:, 0],
            desobj.zs[n0:, 1],
            marker="o",
            facecolors="none",
            edgecolors="blue",
            zorder=2
        )
    
        # Overlay test design points
        for xpt in xr:
            ax[1, mid].scatter(xpt[0], xpt[1], marker="x", color="black", s=100, zorder=3)
    
        # Contour of noise
        cs = ax[1, mid].contourf(
            X1, X2, ng.reshape(nmesh, nmesh),
            cmap=yellow_cmap,
            alpha=0.75,
            zorder=1
        )
    
        ax[1, mid].set_xlabel(r"$x_1$", fontsize=ft)
        ax[1, mid].set_ylabel(r"$x_2$", fontsize=ft)
        ax[1, mid].set_xticks([0, 0.5, 1])
        ax[1, mid].set_yticks([0, 0.5, 1])
    
    if title is not None:
        plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.show()


def plot_horizon_progress(
    path,
    example="SIR",
    method="ivar",
    single_seed=1,
    labels=("target", "adapt"),
    reps=(1, 31),
    figsize=(8, 2),
    colors=("gray", "orange"),
    title=None,
):
 
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    for mid, label in enumerate(labels):
        # Read and filter data
        df = read_data(
            rep0=reps[0],
            repf=reps[1],
            methods=[method],
            examples=[example],
            folderpath=path,
            label=[label],
        )

        df_label = df.loc[df["h"] == label].copy()
        df_label["new_cumsum"] = df_label.groupby("s")["new"].cumsum()

        # Plot for a single seed
        df_seed1 = df_label[df_label["s"] == single_seed]
        ax[0].plot(df_seed1["t"], df_seed1["horizon"], color=colors[mid], linestyle=':')

        # Median horizon across reps
        df_median = df_label.groupby("t", as_index=False)["horizon"].mean()
        ax[0].plot(df_median["t"], df_median["horizon"], color=colors[mid], label=label, linewidth=3)

        # Mean cumulative new per t
        df_cumsum_avg = df_label.groupby("t", as_index=False)["new_cumsum"].mean()
        ax[1].plot(
            df_cumsum_avg["t"],
            df_cumsum_avg["new_cumsum"] / np.arange(1, len(df_cumsum_avg) + 1),
            color=colors[mid],
            linewidth=3,
            label=label,
        )

    # Axis labels and legends
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Horizon")
    ax[0].legend()

    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$n_t / \sum_{i=1}^{n_t} a_i$")
    ax[1].legend()
    plt.tight_layout()
    
    if title is not None:
        plt.savefig(title, dpi=300, bbox_inches="tight")
        
    plt.show()

    
def plot_pairwise_from_output(path, example, ms, n0=90, rep0=5, d=2, label="h=-1"):
    
    for m in ms:
        desobj = read_output(path, example, m, 1, label=label)
        df = pd.DataFrame(desobj.zs[(n0 * rep0):, d:])
        sns.pairplot(df, diag_kind="kde")
        plt.suptitle(f"m = {m}", y=1.02)
        plt.tight_layout()
        plt.show()


def boxplot_alternative(path, methods, horizons, example, ax, figset={}):
    ifLeg, sety = figset.get("ifLeg", True), figset.get("sety", True)
    legend_right = figset.get("legend_right", True)              
    ft = figset.get("ft", 10)
    r0, n0 = figset.get("r0", 5), figset.get("n0", 50)

    lst = []
    for m in methods:
        for hor in horizons:
            for i in np.arange(1, 31):
                desobj = read_output(path, example, m, i, label=hor)
                
                # Eliminate the initial sample
                initial_des = desobj.zs[0:n0, :]
                unique_init, counts_init = np.unique(initial_des, axis=0, return_counts=True)
                
                # Convert unique_init to a set of tuple rows for fast lookup
                unique_init_set = set(map(tuple, unique_init))
                
                count_reps = []
                for xtid, xt in enumerate(desobj.xt):
                    xt_tuple = tuple(xt)
                    if (xt_tuple in unique_init_set) and (desobj.reps[xtid] == r0):
                        continue  # skip this one
                    count_reps.append(desobj.reps[xtid])
                    
                count_reps = np.array(count_reps)
                total = len(count_reps)
                # Count percentages
                percent_1 = np.sum(count_reps == 1) / total * 100
                percent_2 = np.sum(count_reps == 2) / total * 100
                percent_3 = np.sum(count_reps == 3) / total * 100
                percent_4 = np.sum(count_reps == 4) / total * 100
                percent_5 = np.sum(count_reps >= 5) / total * 100
                
                lst.append({
                    "1": percent_1, 
                    "2": percent_2, 
                    "3": percent_3, 
                    "4": percent_4, 
                    "5": percent_5, 
                    "r": i, 
                    "method": m, 
                    "h": hor
                })
    
    df = pd.DataFrame(lst)
    
    # Average over r before plotting
    components = ['1', '2', '3', '4', '5']
    
    df_avg = df.groupby(['h', 'method'])[components].mean().reset_index()
    
    # Prepare for plotting
    h_values = sorted(df_avg['h'].unique())
    methods_u = df_avg['method'].unique()
    num_methods = len(methods_u)
    
    bar_width = 0.1
    group_spacing = 0.1
    hatches = ['/', '\\', 'x', 'o', '.'] 
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    # Create x positions
    x_positions = []
    current_x = 0
    grouped = df_avg.sort_values(['h', 'method']).reset_index(drop=True)
    
    for i, h in enumerate(h_values):
        methods_in_group = grouped[grouped['h'] == h]
        for _ in range(len(methods_in_group)):
            x_positions.append(current_x)
            current_x += bar_width
        current_x += group_spacing
    
    # Plot
    #fig, ax = plt.subplots(figsize=(6, 6))
    bottoms = np.zeros(len(grouped))
    
    # First pass: fill with white so we can hatch over it
    for i, comp in enumerate(components):
        ax.bar(
            x_positions,
            grouped[comp],
            bottom=bottoms,
            width=bar_width,
            color='white',
            edgecolor='black'
        )
        bottoms += grouped[comp].values
    
    # Second pass: hatch overlay with colored edges
    bottoms = np.zeros(len(grouped))
    for i, comp in enumerate(components):
        ax.bar(
            x_positions,
            grouped[comp],
            bottom=bottoms,
            width=bar_width,
            fill=False,
            hatch=hatches[i % len(hatches)],
            edgecolor=colors[i % len(colors)],
            linewidth=1.5,
            label=f'{comp}'
        )
        bottoms += grouped[comp].values
    
    # Method labels on x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped['method'].values, rotation=45, ha='right')
    
    # h group labels below the x-axis
    for i, h in enumerate(h_values):
        group_start = i * (num_methods * bar_width + group_spacing)
        group_mid = group_start + (num_methods - 1) * bar_width / 2
        ax.text(
            group_mid, -0.002 * ax.get_ylim()[1],
            f'h = {h}',
            ha='center',
            va='top',
            fontsize=ft,
            transform=ax.get_xaxis_transform()
        )
        
    if legend_right:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(1.3, 1),
            fancybox=True,
            shadow=True,
            ncol=1,
            fontsize=ft)
    else:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.6),
            fancybox=True,
            shadow=True,
            ncol=3,
            fontsize=ft)

    if sety == False:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Percentage (%)", fontsize=ft)
    ax.tick_params(axis="both", labelsize=ft)
    if ifLeg == False:
        ax.legend().remove()
