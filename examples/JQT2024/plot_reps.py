from PUQ.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plotresult(path, out, ex_name, w, b, rep, method, n0, nf):

    HDlist = []
    TVlist = []
    timelist = []
    for i in range(0, rep):
        design_saved = read_output("", ex_name, method, w, b, i)

        TV = design_saved._info["TV"]
        HD = design_saved._info["HD"]

        TVlist.append(TV[n0:nf])
        HDlist.append(HD[n0:nf])

    avgTV = np.mean(np.array(TVlist), 0)
    sdTV = np.std(np.array(TVlist), 0)
    avgHD = np.mean(np.array(HDlist), 0)
    sdHD = np.std(np.array(HDlist), 0)

    return avgHD, sdHD, avgTV, sdTV


def plot_aggregated(example_name="sinfunc", is_bias=False, rep=30):

    # choose either 'pritam' or 'sinfunc'
    clist = ["b", "r", "g", "m", "y", "c", "pink", "purple"]
    mlist = ["P", "p", "*", "o", "s", "h"]
    linelist = ["-", "--", "-.", ":", "-.", ":"]
    labelsb = [
        r"$\mathcal{A}^y$",
        r"$\mathcal{A}^p$",
        r"$\mathcal{A}^{lhs}$",
        r"$\mathcal{A}^{rnd}$",
    ]
    batch = 1
    worker = 2
    fonts = 18
    # path = '/Users/ozgesurer/Desktop/des_examples/newPUQ/examples/'
    # example_name = 'pritam'
    # is_bias = True
    if example_name == "pritam":
        n0, nf = 30, 180
        if is_bias:
            outs = "pritam_bias"
            method = ["ceivarxbias", "ceivarbias", "lhs", "rnd"]
        else:
            outs = "pritam"
            method = ["ceivarx", "ceivar", "lhs", "rnd"]

    elif example_name == "sinfunc":
        n0, nf = 10, 100
        if is_bias:
            outs = "sinf_bias"
            method = ["ceivarx", "ceivar", "lhs", "rnd"]
        else:
            outs = "sinf"
            method = ["ceivarx", "ceivar", "lhs", "rnd"]

    for metric in ["TV", "HD"]:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        plt.rcParams["figure.autolayout"] = True
        for mid, m in enumerate(method):
            avgPOST, sdPOST, avgPRED, sdPRED = plotresult(
                None, outs, example_name, worker, batch, rep, m, n0=n0, nf=nf
            )
            if metric == "TV":
                axes.plot(
                    np.arange(len(avgPRED)),
                    avgPRED,
                    label=labelsb[mid],
                    color=clist[mid],
                    linestyle=linelist[mid],
                    linewidth=4,
                )
                plt.fill_between(
                    np.arange(len(avgPRED)),
                    avgPRED - 1.96 * sdPRED / rep,
                    avgPRED + 1.96 * sdPRED / rep,
                    color=clist[mid],
                    alpha=0.1,
                )
            elif metric == "HD":
                axes.plot(
                    np.arange(len(avgPOST)),
                    avgPOST,
                    label=labelsb[mid],
                    color=clist[mid],
                    linestyle=linelist[mid],
                    linewidth=4,
                )
                plt.fill_between(
                    np.arange(len(avgPOST)),
                    avgPOST - 1.96 * sdPOST / rep,
                    avgPOST + 1.96 * sdPOST / rep,
                    color=clist[mid],
                    alpha=0.1,
                )
        axes.set_yscale("log")
        # axes.set_xscale('log')
        axes.set_xlabel("# of parameters", fontsize=fonts)

        if metric == "TV":
            axes.set_ylabel(r"${\rm MAD}^y$", fontsize=fonts)
        elif metric == "HD":
            axes.set_ylabel(r"${\rm MAD}^p$", fontsize=fonts)
        axes.tick_params(axis="both", which="major", labelsize=fonts - 5)

        axes.legend(bbox_to_anchor=(1, -0.2), ncol=4, fontsize=fonts, handletextpad=0.1)
        plt.show()
