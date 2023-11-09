import numpy as np
import scipy.stats as sps

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
    
    def pdf(self, theta):
        ncnd   = theta.shape[0]
        thlist = np.ones(ncnd)
        for i in range(self.p):
            thlist *= sps.uniform.pdf(theta[:, i], self.a[i], self.b[i]-self.a[i])
        return np.array(thlist).T


class prior_truncnorm():
    def __init__(self, a, b, loc, scale):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
        self.p = len(a)
        
    def rnd(self, n, seed=None):
        if seed == None:
            pass
        else:
            np.random.seed(seed)
        thlist = []
        for i in range(self.p):
            thlist.append(sps.truncnorm.rvs(a=self.a[i], b=self.b[i], loc=self.loc[i], scale=self.scale[i], size=n))
        return np.array(thlist).T        

    def pdf(self, theta):
        ncnd   = theta.shape[0]
        thlist = np.ones(ncnd)
        for i in range(self.p):
            thlist *= sps.truncnorm.pdf(theta[:, i], a=self.a[i], b=self.b[i], loc=self.loc[i], scale=self.scale[i])
        return np.array(thlist).T
    
    #if seed == None:
    #    pass
    #else:
    #    np.random.seed(seed)
    #sps.truncnorm.rvs(a=-2.5, b=1.5, loc=3, scale=2, size=1000)   
    # 1.5=(6-3)/2, -2.5=(-2-3)/2
    # 3=(6-3)/1, -5=(-2-3)/1
    # 6=(6-3)/0.5, -10=(-2-3)/0.5
    #class prior_norm:                                                                            
    #    def rnd(n):
    #        thlist = []
    #        for i in range(thetalimits.shape[0]):
    #            thlist.append(sps.truncnorm.rvs(a=-10, b=2, loc=3, scale=1, size=n))
    #        return np.array(thlist).T
    
    #    def pdf(thetacnd):
    #        ncnd   = thetacnd.shape[0]
    #        thlist = np.ones(ncnd)
    #        for i in range(thetalimits.shape[0]):
    #            thlist *= sps.truncnorm.pdf(thetacnd[:, i], a=-10, b=2, loc=3, scale=1)
    #        return np.array(thlist).T
    
    #if rnd == True:
    #    thetas = prior_norm.rnd(n)
    #    return thetas
    #else:
    #    thetapdf = prior_norm.pdf(thetacnd)
    #    return thetapdf
    
def prior_dist(dist='uniform'):
    return eval('prior_'+dist)
    
#class prior_norm:
#    def __init__(self, loc, scale):
#        self.data = []
        
#class prior_unif:
#    def __init__(self, a, b):
#        self.data = []
#        self.a = a
#        self.b = b
#
#dist = 'unif'
#cls_pr = eval('prior_'+dist)(2, 3)