import numpy as np
import matplotlib.pyplot as plt
from PUQ.performanceutils.accuracy import get_batched_accuracy


def find_threshold(acc_level, acc_list):
    threshold = [a for a in acc_list if a <= acc_level][0]
    id_thr = np.where(acc_list == threshold)[0][0]
    return threshold, id_thr


def plot_workers(PM, joblist, acqlist):
    wk = np.zeros(PM.worker)
    complete_stage = 0
    total_acq_time = 0
    fig, ax = plt.subplots()
    for jid, job in enumerate(joblist):
        if job.end <= PM.complete_time:
            ax.hlines(
                y=job.workerid + 1, xmin=job.start, xmax=job.end, color="b", linewidth=1
            )
            ax.vlines(
                x=job.start,
                ymin=job.workerid + 1 - 0.25,
                ymax=job.workerid + 1 + 0.25,
                color="b",
                linestyles="dashed",
                linewidth=1,
            )
            ax.vlines(
                x=job.end,
                ymin=job.workerid + 1 - 0.25,
                ymax=job.workerid + 1 + 0.25,
                color="b",
                linestyles="dashed",
                linewidth=1,
            )

            wk[job.workerid] += job.end - job.start

    for ida, acq in enumerate(acqlist):
        if acq.end <= PM.complete_time:
            ax.hlines(y=0, xmin=acq.start, xmax=acq.end, color="r", linewidth=1)
            ax.vlines(
                x=acq.start,
                ymin=0 - 0.25,
                ymax=0 + 0.25,
                linestyles="dashed",
                color="r",
                linewidth=1,
            )
            ax.vlines(
                x=acq.end,
                ymin=0 - 0.25,
                ymax=0 + 0.25,
                linestyles="dashed",
                color="r",
                linewidth=1,
            )
            complete_stage += 1
            total_acq_time += acq.end - acq.start

    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Worker", fontsize=18)
    txt1 = "Completion:" + str(np.round(PM.complete_time, 1)) + "\n"
    txt2 = "# of jobs:" + str(np.round(PM.complete_no, 1)) + "\n"

    txt3 = "# of stages:" + str(PM.completed_stage) + "\n"
    txt4 = "Acq time:" + str(np.round(PM.total_acq_time, 2)) + "\n"

    txt5 = "mean (idle):" + str(np.round(PM.complete_time - np.mean(wk))) + "\n"
    txt6 = "std (idle):" + str(np.round(np.std(wk)))
    plt.figtext(
        0.95,
        0.2,
        txt1 + txt2 + txt3 + txt4 + txt5 + txt6,
        fontname="DejaVu Sans Mono",
        fontsize=18,
    )
    # plt.xlim(0, sorted(endjob)[-1])
    plt.title(str(PM.batch) + "," + str(PM.worker))
    plt.show()


def plot_acc(axes, n, acclevel, rlist, labellist, logscale=False, fontsize=18, n0=0):

    clist = ["b", "r", "g", "m", "y", "c"]
    mlist = ["P", "o", "*", "s", "p", "h"]
    linelist = ["-", "--", "-.", ":", "-.", ":"]
    for accid, res in enumerate(rlist):
        axes.plot(
            np.arange(1, n + 1),
            res.acc,
            linestyle=linelist[accid],
            linewidth=5.0,
            label=labellist[accid],
            color=clist[accid],
        )
        axes.vlines(
            x=res.complete_no,
            ymin=0,
            ymax=res.acc_threshold,
            linewidth=5,
            color=clist[accid],
            linestyles=(0, (2, 5)),
        )

    axes.hlines(y=acclevel, xmin=0, xmax=n, linewidth=2, color="k")
    axes.set_xlabel("# of parameters", fontsize=fontsize)
    axes.set_ylabel("Error", fontsize=fontsize)
    axes.tick_params(axis="both", which="major", labelsize=fontsize - 5)
    axes.set_xlim([n0, n])
    axes.set_ylim(0, 1.1)
    if logscale:
        axes.set_xscale("log")
        axes.set_yscale("log")


def plot_acqtime(
    axes, n, acclevel, rlist, labellist, logscale=False, fontsize=18, ind=False, n0=0
):
    clist = [
        "blue",
        "red",
        "green",
        "magenta",
        "orange",
        "dimgrey",
        "lime",
        "dimgrey",
        "lime",
    ]

    for accid, res in enumerate(rlist):
        if ind:
            axes.plot(
                np.arange(1, n + 1),
                res.gentime,
                label=labellist[accid],
                color=clist[accid],
            )
        else:
            axes.plot(
                np.arange(1, n + 1),
                np.cumsum(res.gentime),
                label=labellist[accid],
                color=clist[accid],
            )

    axes.set_xlabel("# of parameters", fontsize=fontsize)
    axes.set_ylabel("Acquisition time", fontsize=fontsize)
    axes.tick_params(axis="both", which="major", labelsize=fontsize - 5)
    axes.set_xlim([n0, n])
    if logscale:
        axes.set_xscale("log")
        axes.set_yscale("log")


def plot_endtime(
    axes, n, acclevel, rlist, labellist, worker, logscale=False, fontsize=18
):

    clist = ["b", "r", "g", "m", "y", "c"]
    mlist = ["P", "o", "*", "s", "p", "h"]
    linelist = ["-", "--", "-.", ":", "-.", ":"]

    minworker = []
    maxtime = 0
    for endid, res in enumerate(rlist):
        endtime = [job.end for job in res.sortedjobs]
        minworker.append(endtime[worker])
        if maxtime < np.max(endtime):
            maxtime = np.max(endtime)
        axes.plot(
            np.arange(1, n + 1),
            endtime,
            linestyle=linelist[endid],
            linewidth=5.0,
            label=labellist[endid],
            color=clist[endid],
        )
        axes.vlines(
            x=res.complete_no,
            ymin=0,
            ymax=endtime[res.complete_no],
            linewidth=5,
            color=clist[endid],
            linestyles=(0, (2, 5)),
        )
        axes.hlines(
            y=endtime[res.complete_no],
            xmin=0,
            xmax=res.complete_no,
            linewidth=5,
            color=clist[endid],
            linestyles=(0, (2, 5)),
        )

    axes.set_xlabel("# of parameters", fontsize=fontsize)
    axes.set_ylabel("Wall-clock time", fontsize=fontsize)
    if logscale:
        axes.set_xscale("log")
        axes.set_yscale("log")
    axes.set_xlim([worker, n])
    axes.set_ylim([np.min(minworker), maxtime])
    axes.tick_params(axis="both", which="major", labelsize=fontsize - 5)


def plot_errorend(
    axes, n, acclevel, rlist, labellist, worker, logscale=False, fontsize=18
):
    clist = ["b", "r", "g", "m", "y", "c"]
    mlist = ["P", "o", "*", "s", "p", "h"]
    linelist = ["-", "--", "-.", ":", "-.", ":"]

    minworker = []
    maxtime = 0
    minacc = 10
    for endid, res in enumerate(rlist):
        endtime = [job.end for job in res.sortedjobs]
        minworker.append(endtime[worker])
        if maxtime < np.max(endtime):
            maxtime = np.max(endtime)
        if minacc > np.min(res.acc):
            minacc = np.min(res.acc)
        axes.plot(
            endtime,
            res.acc,
            linestyle=linelist[endid],
            linewidth=5.0,
            label=labellist[endid],
            color=clist[endid],
        )
        axes.vlines(
            x=endtime[res.complete_no],
            ymin=0,
            ymax=acclevel,
            linewidth=5,
            color=clist[endid],
            linestyles=(0, (2, 5)),
        )

    axes.hlines(y=acclevel, xmin=0, xmax=maxtime, linewidth=2, color="k")

    if logscale:
        axes.set_xscale("log")
        axes.set_yscale("log")
    axes.set_xlim([np.min(minworker), maxtime])
    axes.set_ylim([minacc, np.max([res.acc[worker] for res in rlist])])
    axes.set_xlabel("Wall-clock time", fontsize=fontsize)
    axes.set_ylabel("Error", fontsize=fontsize)
    axes.tick_params(axis="both", which="major", labelsize=fontsize - 5)
