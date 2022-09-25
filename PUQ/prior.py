import numpy as np
import scipy.stats as sps

def prior_uniform(n, thetalimits, seed=None):
    """Generate and return n parameters for the test function."""
    if seed == None:
        pass
    else:
        np.random.seed(seed)
        
    class prior_uniform:                                                                            
        def rnd(n):
            thlist = []
            for i in range(thetalimits.shape[0]):
                thlist.append(sps.uniform.rvs(thetalimits[i][0], thetalimits[i][1]-thetalimits[i][0], size=n))
            return np.array(thlist).T
        
    thetas = prior_uniform.rnd(n)
    return thetas

def prior_dist(dist='uniform'):
    return eval('prior_'+dist)
    