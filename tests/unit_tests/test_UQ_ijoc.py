import numpy as np
import pytest
from contextlib import contextmanager
from PUQ.design import designer
from PUQ.prior import prior_dist
import scipy.stats as sps


class himmelblau:
    def __init__(self):

        self.data_name = "himmelblau"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-5, 5], [-5, 5]])
        self.obsvar = np.array([[1]], dtype="float64")
        self.real_data = np.array([[1]], dtype="float64")
        self.out = [("f", float)]
        self.p = 2
        self.d = 1
        self.x = np.arange(0, self.d)[:, None]
        self.real_x = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):

        theta1 = self.truelimits[0][0] + theta1 * (
            self.truelimits[0][1] - self.truelimits[0][0]
        )
        theta2 = self.truelimits[1][0] + theta2 * (
            self.truelimits[1][1] - self.truelimits[1][0]
        )
        f = (theta1**2 + theta2 - 11) ** 2 + (theta1 + theta2**2 - 7) ** 2
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the himmelblau function
        """

        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])
        return H_o, persis_info


example = "himmelblau"
cls_data = eval(example)()

# # # Create a mesh for test set # # #
xpl = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 50)
ypl = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 50)
Xpl, Ypl = np.meshgrid(xpl, ypl)
th = np.vstack([Xpl.ravel(), Ypl.ravel()])
setattr(cls_data, "theta", th.T)

ftest = np.zeros(2500)
for tid, t in enumerate(th.T):
    ftest[tid] = cls_data.function(t[0], t[1])
thetatest = th.T
ptest = np.zeros(thetatest.shape[0])
for i in range(ftest.shape[0]):
    mean = ftest[i]
    rnd = sps.multivariate_normal(mean=mean, cov=cls_data.obsvar)
    ptest[i] = rnd.pdf(cls_data.real_data)

test_data = {"theta": thetatest, "f": ftest, "p": ptest, "p_prior": 1}

# # # # # # # # # # # # # # # # # # # # #
prior_func = prior_dist(dist="uniform")(
    a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]
)

n_init = 10
s = 0
thetainit = prior_func.rnd(n_init, s)
finit = np.zeros(n_init)
for tid, t in enumerate(thetainit):
    finit[tid] = cls_data.function(t[0], t[1])
test_data["thetainit"] = thetainit
test_data["finit"] = finit[None, :]


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "input1,expectation",
    [("ei", does_not_raise()), ("hybrid_ei", does_not_raise())],
)
def test_none_input(input1, expectation):
    with expectation:
        assert (
            designer(
                data_cls=cls_data,
                method="SEQCALOPT",
                args={
                    "mini_batch": 1,
                    "nworkers": 2,
                    "AL": input1,
                    "seed_n0": 0,
                    "prior": prior_func,
                    "data_test": test_data,
                    "max_evals": 50,
                    "candsize": 100,
                    "refsize": 100,
                    "believer": 0,
                },
            )
            is not None
        )
