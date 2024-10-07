import numpy as np


class sinfunc:
    def __init__(self):
        self.data_name = "sinfunc"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta = np.array([np.pi / 5])
        self.out = [("f", float)]
        self.d = 1
        self.p = 2
        self.dx = 1
        self.x = None
        self.real_data = None
        self.sigma2 = 0.2**2
        self.nodata = True

    def function(self, x, theta):
        f = np.sin(10 * x - 5 * theta)
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1])
        return H_o, persis_info

    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))

        np.random.seed(seed)
        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = self.genobsdata(x, isbias)

        self.real_data = np.array([fevals], dtype="float64")

    def genobsdata(self, x, isbias=False):
        if isbias:
            return (
                self.function(x[0], self.true_theta[0])
                + self.bias(x[0])
                + np.random.normal(0, np.sqrt(self.sigma2), 1)
            )
        else:
            return self.function(x[0], self.true_theta[0]) + np.random.normal(
                0, np.sqrt(self.sigma2), 1
            )

    def bias(self, x):
        return 1 - (1 / 3) * x - (2 / 3) * (x**2)


class pritam:
    def __init__(self):
        self.data_name = "pritam"
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1]])
        self.true_theta = np.array([0.5])
        self.out = [("f", float)]
        self.d = 1
        self.p = 3
        self.x = None
        self.real_data = None
        self.dx = 2
        self.sigma2 = 0.5**2
        self.nodata = True

    def function(self, x1, x2, theta1):
        f = (30 + 5 * x1 * np.sin(5 * x1)) * (6 * theta1 + 1 + np.exp(-5 * x2))
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(H["thetas"][0][0], H["thetas"][0][1], H["thetas"][0][2])
        return H_o, persis_info

    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))

        np.random.seed(seed)
        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = self.genobsdata(x, isbias)

        self.real_data = np.array([fevals], dtype="float64")

    def genobsdata(self, x, isbias=False):
        if isbias:
            return (
                self.function(x[0], x[1], self.true_theta[0])
                + self.bias(x[0], x[1])
                + np.random.normal(0, np.sqrt(self.sigma2), 1)
            )
        else:
            return self.function(x[0], x[1], self.true_theta[0]) + np.random.normal(
                0, np.sqrt(self.sigma2), 1
            )

    def bias(self, x1, x2):
        return -50 * (np.exp(-0.2 * x1 - 0.1 * x2))


class highdim2:
    def __init__(self):
        self.data_name = "highdim"
        self.thetalimits = np.array(
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ]
        )
        self.true_theta = None
        self.out = [("f", float)]
        self.d = 1
        self.p = 12
        self.x = None
        self.real_data = None
        self.dx = None
        self.sigma2 = None  # 1#5#10
        self.nodata = True

    def function(
        self,
        theta1,
        theta2,
        theta3,
        theta4,
        theta5,
        theta6,
        theta7,
        theta8,
        theta9,
        theta10,
        theta11,
        theta12,
    ):

        a = -5
        b = 10
        if self.dx == 2:
            # print(self.dx)
            theta3 = a + b * theta3
            theta4 = a + b * theta4
            theta5 = a + b * theta5
            theta6 = a + b * theta6
            theta7 = a + b * theta7
            theta8 = a + b * theta8
            theta9 = a + b * theta9
            theta10 = a + b * theta10
            theta11 = a + b * theta11
            theta12 = a + b * theta12
        elif self.dx == 6:
            # print(self.dx)
            theta7 = a + b * theta7
            theta8 = a + b * theta8
            theta9 = a + b * theta9
            theta10 = a + b * theta10
            theta11 = a + b * theta11
            theta12 = a + b * theta12
        elif self.dx == 10:
            # print(self.dx)
            theta11 = a + b * theta11
            theta12 = a + b * theta12

        if self.x.shape[1] == 2:
            f = np.sqrt(np.abs(theta1 + theta2)) * np.power(
                theta3
                + theta4
                + theta5
                + theta6
                + theta7
                + theta8
                + theta9
                + theta10
                + theta11
                + theta12,
                2,
            )
            # f = np.sqrt(np.abs(theta1 + theta2))*((theta3 + theta4)**2 + (theta5 + theta6)**2 + (theta7 + theta8)**2 + (theta9 + theta10)**2 + (theta11 + theta12)**2)
        elif self.x.shape[1] == 6:
            f = np.sqrt(
                np.abs(theta1 + theta2 + theta3 + theta4 + theta5 + theta6)
            ) * np.power(theta7 + theta8 + theta9 + theta10 + theta11 + theta12, 2)
            # f = np.sqrt(np.abs(theta1 + theta2 + theta3 + theta4 + theta5 + theta6))*((theta7 + theta8)**2 + (theta9 + theta10)**2 + (theta11 + theta12)**2)
        elif self.x.shape[1] == 10:
            f = np.sqrt(
                np.abs(
                    theta1
                    + theta2
                    + theta3
                    + theta4
                    + theta5
                    + theta6
                    + theta7
                    + theta8
                    + theta9
                    + theta10
                )
            ) * np.power(theta11 + theta12, 2)
            # f = np.sqrt(np.abs(theta1 + theta2 + theta3 + theta4 + theta5 + theta6 + theta7 + theta8 + theta9 + theta10))*((theta11 + theta12)**2)
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function = sim_specs["user"]["function"]
        H_o = np.zeros(1, dtype=sim_specs["out"])
        H_o["f"] = function(
            H["thetas"][0][0],
            H["thetas"][0][1],
            H["thetas"][0][2],
            H["thetas"][0][3],
            H["thetas"][0][4],
            H["thetas"][0][5],
            H["thetas"][0][6],
            H["thetas"][0][7],
            H["thetas"][0][8],
            H["thetas"][0][9],
            H["thetas"][0][10],
            H["thetas"][0][11],
        )
        return H_o, persis_info

    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.dx = x.shape[1]

        if self.dx == 2:
            self.sigma2 = 25
        elif self.dx == 6:
            self.sigma2 = 5
        elif self.dx == 10:
            self.sigma2 = 1

        self.true_theta = np.repeat(0.5, self.p - self.dx)
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, x.shape[0]))

        np.random.seed(seed)
        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = self.genobsdata(x, isbias)

        self.real_data = np.array([fevals], dtype="float64")

    def genobsdata(self, x, isbias=False):

        if len(x) == 2:
            print("Obs:", len(x))
            return self.function(
                x[0],
                x[1],
                self.true_theta[0],
                self.true_theta[1],
                self.true_theta[2],
                self.true_theta[3],
                self.true_theta[4],
                self.true_theta[5],
                self.true_theta[6],
                self.true_theta[7],
                self.true_theta[8],
                self.true_theta[9],
            ) + np.random.normal(0, np.sqrt(self.sigma2), 1)
        elif len(x) == 6:
            print("Obs:", len(x))
            return self.function(
                x[0],
                x[1],
                x[2],
                x[3],
                x[4],
                x[5],
                self.true_theta[0],
                self.true_theta[1],
                self.true_theta[2],
                self.true_theta[3],
                self.true_theta[4],
                self.true_theta[5],
            ) + np.random.normal(0, np.sqrt(self.sigma2), 1)

        elif len(x) == 10:
            print("Obs:", len(x))
            return self.function(
                x[0],
                x[1],
                x[2],
                x[3],
                x[4],
                x[5],
                x[6],
                x[7],
                x[8],
                x[9],
                self.true_theta[0],
                self.true_theta[1],
            ) + np.random.normal(0, np.sqrt(self.sigma2), 1)
