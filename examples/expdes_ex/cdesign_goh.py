import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots import plot_EIVAR, plot_LHS, obsdata, fitemu, create_test, gather_data
from smt.sampling_methods import LHS
from test_funcs import gohbostos


s = 1
cls_data = gohbostos()
cls_data.realdata(s)

#x1 = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 5)
#x2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 5)

#X1, X2 = np.meshgrid(x1, x2)
#XS = np.vstack([X1.ravel(), X2.ravel()])

t1 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 10)
t2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 10)

T1, T2 = np.meshgrid(t1, t2)
TS = np.vstack([T1.ravel(), T2.ravel()])

XT = np.zeros((2500, 4))
f = np.zeros((2500))
thetamesh = np.zeros((2500, 2))
k = 0
TS[0][0] = 0.2
TS[1][0] = 0.1
for j in range(100):
    for i in range(25):
        XT[k, :] = np.array([cls_data.real_x[i, 0], cls_data.real_x[i, 1], TS[0][j], TS[1][j]])
        f[k] = cls_data.function(cls_data.real_x[i, 0], cls_data.real_x[i, 1], TS[0][j], TS[1][j])
        k += 1
    
    thetamesh[j, :] = np.array([TS[0][j], TS[1][j]])
    
ftest = f.reshape(100, 25)
ptest = np.zeros(100)

for j in range(100):
    rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
    ptest[j] = rnd.pdf(cls_data.real_data)
    

test_data = {'theta': XT, 
             'f': ftest,
             'p': ptest,
             'th': thetamesh[:, None],    
             'xmesh': 0,
             'p_prior': 1} 

prior_func      = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]) 


print(thetamesh[np.argmax(ptest), :])


ninit = 10
al_unimodal = designer(data_cls=cls_data, 
                       method='SEQCOMPDES', 
                       args={'mini_batch': 1, 
                             'n_init_thetas': ninit,
                             'nworkers': 2, 
                             'AL': 'eivar_exp',
                             'seed_n0': s,
                             'prior': prior_func,
                             'data_test': test_data,
                             'max_evals': 50,
                             'type_init': None,
                             'unknown_var': False,
                             'design': False})