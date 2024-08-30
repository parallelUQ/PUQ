import numpy as np
import pytest
from contextlib import contextmanager
from PUQ.design import designer
from PUQ.prior import prior_dist


class unimodal:
    def __init__(self):
        self.data_name = "unimodal"
        self.thetalimits = np.array([[-4, 4], [-4, 4]])
        self.obsvar = np.array([[4]], dtype="float64")
        self.real_data = np.array([[-6]], dtype="float64")
        self.out = [("f", float)]
        self.d = 1
        self.p = 2
        self.x = np.arange(0, self.d)[:, None]
        self.real_x = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):
        """
        Wraps the unimodal function
        """
        thetas = np.array([theta1, theta2]).reshape((1, 2))
        S = np.array([[1, 0.5], [0.5, 1]])
        f = (thetas @ S) @ thetas.T
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the simulator
        """
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


cls_unimodal = unimodal()
prior_func = prior_dist(dist="uniform")(
    a=cls_unimodal.thetalimits[:, 0], b=cls_unimodal.thetalimits[:, 1]
)


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "input1,expectation",
    [
        ("maxvar", does_not_raise()),
        ("maxexp", does_not_raise()),
        ("rnd", does_not_raise()),
    ],
)
def test_none_input(input1, expectation):
    with expectation:
        assert (
            designer(
                data_cls=cls_unimodal,
                method="SEQCAL",
                args={
                    "mini_batch": 1,
                    "n_init_thetas": 10,
                    "nworkers": 2,
                    "AL": input1,
                    "seed_n0": 1,
                    "prior": prior_func,
                    "data_test": None,
                    "max_evals": 60,
                    "type_init": None,
                },
            )
            is not None
        )
