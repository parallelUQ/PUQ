"""Tests whether the we obtain a design for a one-dimensional deterministic simulation model"""

import sys

sys.path.append("./")
import numpy as np
from smt.sampling_methods import LHS
from PUQ.designmethods.sequential_1d_deterministic import sequential_design


class sinfunc:
    def __init__(self):
        self.data_name = "sinfunc"
        self.zlim = np.array([[0, 1], [0, 1]])
        self.true_theta = np.array([np.pi / 5])
        self.out = [("f", float)]
        self.d = 1
        self.p = 2
        self.dx = 1
        self.dt = 1
        self.x = None
        self.real_data = None
        self.sigma2 = 0.2**2
        self.nodata = True

    def function(self, x, theta):
        f = np.sin(10 * x - 5 * theta)
        return f

    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.d = len(x)
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))

        np.random.seed(seed)
        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = self.genobsdata(x, isbias)

        self.real_data = np.array([fevals], dtype="float64")

    def genobsdata(self, x, isbias=False):
        return self.function(x[0], self.true_theta[0]) + np.random.normal(
            0, np.sqrt(self.sigma2), 1
        )


cex = sinfunc()
dt = len(cex.true_theta)
x_obs = np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None]
cex.realdata(x=x_obs, seed=1)


# Generate initial sample
n0, nmax = 10, 30
sampling = LHS(xlimits=cex.zlim, random_state=1)
z0 = sampling(n0)
f0 = np.array([cex.function(z0[i, 0], z0[i, 1]) for i in range(n0)])


def test_build_design():

    # Generate design
    des_obj = sequential_design(cex)
    des_obj.build_design(
        z0=z0,
        f0=f0[:, None],
        T=nmax,
        af="ivar",
        args={"nL": 200, "seed": 1, "integral": "importance"},
    )

    assert des_obj.zs.shape == (n0 + nmax, 2)
    assert des_obj.fs.shape == (n0 + nmax, 1)


if __name__ == "__main__":
    test_build_design()
