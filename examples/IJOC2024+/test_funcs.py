import numpy as np
from threading import Event


def artificial_time(persis_info, sim_specs):
    rand_stream = persis_info["rand_stream"]
    run_time = rand_stream.normal(0.1, 0.1, 1)
    if run_time[0] < 0.01:
        r = 0.01
    else:
        r = run_time[0]
    Event().wait(r)


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
        artificial_time(persis_info, sim_specs)
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


class holder:
    def __init__(self):

        self.data_name = "holder"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-10, 10], [-10, 10]])
        self.obsvar = np.array([[50]], dtype="float64")
        self.real_data = np.array([[-19.208502567767606]], dtype="float64")
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
        f = -np.abs(
            np.sin(theta1)
            * np.cos(theta2)
            * np.exp(np.abs(1 - (np.sqrt(theta1**2 + theta2**2) / np.pi)))
        )
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the holder function
        """
        artificial_time(persis_info, sim_specs)
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


class ackley:
    def __init__(self):

        self.data_name = "ackley"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-5, 5], [-5, 5]])
        self.obsvar = np.array([[10]], dtype="float64")
        self.real_data = np.array([[0]], dtype="float64")
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
        f = (
            -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (theta1**2 + theta2**2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * theta1) + np.cos(2 * np.pi * theta2)))
            + np.e
            + 20
        )

        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the ackley function
        """
        artificial_time(persis_info, sim_specs)
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


class easom:
    def __init__(self):

        self.data_name = "easom"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-10, 10], [-10, 10]])
        self.obsvar = np.array([[10]], dtype="float64")
        self.real_data = np.array([[-1]], dtype="float64")
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
        f = (
            -np.cos(theta1)
            * np.cos(theta2)
            * np.exp(-((theta1 - np.pi) ** 2 + (theta2 - np.pi) ** 2))
        )
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the easom function
        """
        artificial_time(persis_info, sim_specs)
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


class sphere:
    def __init__(self):

        self.data_name = "sphere"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-5, 5], [-5, 5]])
        self.obsvar = np.array([[10]], dtype="float64")
        self.real_data = np.array([[0]], dtype="float64")
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
        f = theta1**2 + theta2**2
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sphere function
        """
        artificial_time(persis_info, sim_specs)
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info


class matyas:
    def __init__(self):

        self.data_name = "matyas"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-10, 10], [-10, 10]])
        self.obsvar = np.array([[10]], dtype="float64")
        self.real_data = np.array([[0]], dtype="float64")
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
        f = 0.26 * (theta1**2 + theta2**2) - 0.48 * theta1 * theta2
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the matyas function
        """
        artificial_time(persis_info, sim_specs)
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])

        return H_o, persis_info
