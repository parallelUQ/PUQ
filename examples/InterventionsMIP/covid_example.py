from main_deterministic import runfunction
import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np
    
class prior_uniform():
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.p = len(a)

    def rnd(self, n, seed=None):
        if seed == None:
            pass
        else:
            np.random.seed(seed)
        thlist = []
        for i in range(self.p):
            thlist.append(sps.uniform.rvs(self.a[i], self.b[i]-self.a[i], size=n))
        return np.array(thlist).T

priorp = prior_uniform([1, 0.1, 1, 1], [5, 5, 7, 7])
rndvalue = priorp.rnd(10)
rndvalue[9, 0], rndvalue[9, 1], rndvalue[9, 2], rndvalue[9, 3] = 2.9, 0.66, 4, 4

for i in range(10):  
    hosp_ad, daily_ad_benchmark = runfunction(rndvalue[i, :])
    
    
    
    #plt.plot(hosp_benchmark)
    #plt.scatter(np.arange(0, len(real_hosp)), real_hosp)
    #plt.show()
    
    #plt.plot(icu_benchmark)
    #plt.scatter(np.arange(0, len(real_icu)), real_icu)
    #plt.show()
    
    
    plt.plot(daily_ad_benchmark)
    plt.scatter(np.arange(0, len(hosp_ad)), hosp_ad)
    plt.show()   