import numpy as np
import scipy.stats as sps


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


class banana:
    def __init__(self):
        self.data_name = "banana"
        self.thetalimits = np.array([[-20, 20], [-10, 5]])
        self.obsvar = np.array([[10**2, 0], [0, 1]])
        self.real_data = np.array([[1, 3]], dtype="float64")
        self.out = [("f", float, (2,))]
        self.p = 2
        self.d = 2
        self.x = np.arange(0, self.d)[:, None]
        self.real_x = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):
        f = np.array([theta1, theta2 + 0.03 * theta1**2])
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


class unidentifiable:
    def __init__(self):
        self.data_name = "unidentifiable"
        self.thetalimits = np.array([[-8, 8], [-8, 8]])
        self.obsvar = np.array([[1 / 0.01, 0], [0, 1]])
        self.real_data = np.array([[0, 0]], dtype="float64")
        self.out = [("f", float, (2,))]
        self.d = 2
        self.p = 2
        self.x = np.arange(0, self.d)[:, None]
        self.real_x = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):
        f = np.array([theta1, theta2])
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the unidentifiable function
        """
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


class bimodal:
    def __init__(self):
        self.data_name = "bimodal"
        self.thetalimits = np.array([[-6, 6], [-4, 8]])
        self.obsvar = np.array([[1 / np.sqrt(0.2), 0], [0, 1 / np.sqrt(0.75)]])
        self.real_data = np.array([[0, 2]], dtype="float64")
        self.out = [("f", float, (2,))]
        self.d = 2
        self.p = 2
        self.x = np.arange(0, self.d)[:, None]
        self.real_x = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):
        f = np.array([theta2 - theta1**2, theta2 - theta1])
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the bimodal function
        """
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


def create_test_data(al_test, cls_func):
    ftest = al_test._info["f"]
    thetatest = al_test._info["theta"]
    ptest = np.zeros(thetatest.shape[0])

    if cls_func.data_name == "unimodal":
        ptest = sps.norm.pdf(cls_func.real_data - ftest, 0, np.sqrt(cls_func.obsvar))
    else:
        for i in range(ftest.shape[0]):
            mean = ftest[i, :]
            rnd = sps.multivariate_normal(mean=mean, cov=cls_func.obsvar)
            ptest[i] = rnd.pdf(cls_func.real_data)

    test_data = {"theta": thetatest, "f": ftest, "p": ptest, "p_prior": 1}

    return test_data