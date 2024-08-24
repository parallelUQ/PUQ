import numpy as np
import pytest
from contextlib import contextmanager
from PUQ.design import designer
from PUQ.prior import prior_dist
from test_funcs import himmelblau



example = 'himmelblau'
cls_data = eval(example)()

# # # Create a mesh for test set # # # 
xpl = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 50)
ypl = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
setattr(cls_data, 'theta', th.T)

ftest = np.zeros(2500)
for tid, t in enumerate(th.T):
    ftest[tid] = cls_data.function(t[0], t[1])
thetatest = th.T 
ptest = np.zeros(thetatest.shape[0])
for i in range(ftest.shape[0]):
    mean = ftest[i] 
    rnd = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
    ptest[i] = rnd.pdf(cls_data.real_data)
       
test_data = {'theta': thetatest, 
             'f': ftest,
             'p': ptest,
             'p_prior': 1} 

 # # # # # # # # # # # # # # # # # # # # # 
prior_func = prior_dist(dist='uniform')(a=cls_data.thetalimits[:, 0], 
                                        b=cls_data.thetalimits[:, 1])

@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "input1,expectation",
    [
     ('ei', does_not_raise()),
     ('eivar', does_not_raise()),
     ('rnd', does_not_raise()),
     ],
    )
def test_none_input(input1,  expectation):
    with expectation:
        assert designer(data_cls=cls_data, 
                              method='SEQCALOPT', 
                              args={'mini_batch': 1, 
                                    'nworkers': 2,
                                    'AL': input1, 
                                    'seed_n0': 0,
                                    'prior': prior_func,
                                    'data_test': test_data,
                                    'max_evals': 50,
                                    'candsize': 100, 
                                    'refsize': 100,
                                    'believer': 0}) is not None