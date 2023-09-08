from PUQ.designmethods.utils import parse_arguments, save_output, read_output
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np

def plotresult(path, out, ex_name, w, b, rep, method, n0, nf):

 
    avgtime = 0
    avgAE = 0
    avgTV = 0
    avgbestTV = 0
    
    AElist = []
    TVlist = []
    timelist = []
    for i in range(1, 1+rep):
        design_saved = read_output(path + out + '/', ex_name, method, w, b, i)
 
        #theta    = design_saved._info['theta']
        #plt.scatter(theta[:, 0], theta[:, 1])
        #plt.show()
        TV       = design_saved._info['TV']
        AE       = design_saved._info['AE']
        time     = design_saved._info['time']

        if method == 'rnd':
            time = np.repeat(0.1, len(time))
        
        bestTV = np.zeros(TV.shape)
        for i in range(len(TV)):
            bestTV[i] = np.min(TV[0:(i+1)])
            

        AElist.append(AE[n0:nf])
        timelist.append(time[n0:nf])
        TVlist.append(bestTV[n0:nf])
        #avgtime += time
        #avgAE += AE
        #avgTV += TV
        #avgbestTV += bestTV
        #TVlist.append(bestTV)

    avgtime = np.mean(np.array(timelist), 0)
    avgAE   = np.mean(np.array(AElist), 0)
    avgTV   = np.mean(np.array(TVlist), 0)
        


    return avgAE, avgtime, avgTV