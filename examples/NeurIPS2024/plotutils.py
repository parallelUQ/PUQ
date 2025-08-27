from PUQ.designmethods.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plotresult(path, out, ex_name, w, b, rep, method, n0, nf):

    AElist = []
    TVlist = []
    timelist = []
    for i in range(1, 1 + rep):
        design_saved = read_output(path + out + "/", ex_name, method, w, b, i)

        TV = design_saved._info["TV"]
        AE = design_saved._info["AE"]
        time = design_saved._info["time"]

        if method == "rnd":
            time = np.repeat(0.1, len(time))

        bestTV = np.zeros(TV.shape)
        for i in range(len(TV)):
            bestTV[i] = np.min(TV[0 : (i + 1)])

        AElist.append(AE[n0:nf])
        timelist.append(time[n0:nf])
        TVlist.append(bestTV[n0:nf])

    avgtime = np.mean(np.array(timelist), 0)
    avgAE = np.mean(np.array(AElist), 0)
    avgTV = np.mean(np.array(TVlist), 0)

    return avgAE, avgtime, avgTV


def plotparams(path, out, ex_name, w, b, rep, method, n0, nf, thetalim):

    lst = []
    for i in range(1, 1 + rep):
        design_saved = read_output(path + out + "/", ex_name, method, w, b, i)
        lst.append(design_saved._info["AE"][nf])
        # print(design_saved._info['AE'][-1])
        theta = design_saved._info["theta"]
        plt.scatter(theta[:, 0], theta[:, 1])
        plt.xlim(thetalim[0][0], thetalim[0][1])
        plt.ylim(thetalim[1][0], thetalim[1][1])
        plt.show()

    print(len(design_saved._info["AE"]))
    print(np.mean(lst))
