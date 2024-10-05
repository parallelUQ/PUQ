import numpy as np
import dill as pickle
from PUQ.designmethods.utils import read_output


def get_rep_data(s, w, b, rep, filename, method):

    avgtime = 0
    avgAE = 0
    avgTV = 0
    avgbestTV = 0

    AElist = []
    TVlist = []
    timelist = []
    for i in range(1, 1 + rep):
        design_saved = read_output(filename, s, method, w, b, i)

        theta_al = design_saved._info["theta"]
        TV = design_saved._info["TV"]
        HD = design_saved._info["HD"]
        AE = design_saved._info["AE"]
        time = design_saved._info["time"]

        if method == "rnd":
            time = np.repeat(0.1, len(time))

        bestTV = np.zeros(TV.shape)
        for i in range(len(TV)):
            bestTV[i] = np.min(TV[0 : (i + 1)])

        AElist.append(AE)
        timelist.append(time)
        TVlist.append(bestTV)

        avgtime += time
        avgAE += AE
        avgTV += TV
        avgbestTV += bestTV

    avgtime = np.mean(np.array(timelist), 0)
    sdtime = np.std(np.array(timelist), 0) / np.sqrt(30)

    avgAE = np.mean(np.array(AElist), 0)
    sdAE = np.std(np.array(AElist), 0) / np.sqrt(30)

    avgTV = np.mean(np.array(TVlist), 0)
    sdTV = np.std(np.array(TVlist), 0) / np.sqrt(30)

    return avgAE[10:], avgtime[20:]
