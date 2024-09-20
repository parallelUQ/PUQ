import os
from generate_test_data import generate_test_data
import numpy as np
import matplotlib.pyplot as plt
from PUQ.prior import prior_dist
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments
import time

if __name__ == "__main__":

    class bfrescox:
        def __init__(self):
            self.data_name = "bfrescox"
            self.thetalimits = np.array(
                [
                    [40, 60],  # V
                    [0.7, 1.2],  # r
                    # [0.5, 0.8], # a
                    [2.5, 4.5],
                ]
            )  # Ws
            # [0.5, 1.5],
            # [0.1, 0.4]])

            self.d = 15
            self.p = 3
            self.x = np.arange(0, self.d)[:, None]
            self.real_x = np.arange(0, self.d)[:, None]
            self.real_data = np.log(
                np.array(
                    [
                        [
                            1243,
                            887.7,
                            355.5,
                            111.5,
                            26.5,
                            10.4,
                            8.3,
                            7.3,
                            17.2,
                            37.6,
                            48.7,
                            38.9,
                            32.4,
                            36.4,
                            61.9,
                        ]
                    ],
                    dtype="float64",
                )
            )
            self.obsvar = np.diag(np.repeat(0.1, 15))
            self.out = [("f", float, (self.d,))]

        def generate_input_file(self, parameter_values):

            file = "48Ca_template.in"
            with open(file) as f:
                content = f.readlines()
            no_p = 0
            for idx, line in enumerate(content):
                if "XXXXX" in line:
                    no_param = line.count("XXXXX")
                    line_temp = line
                    for i in range(no_param):
                        line_temp = line_temp.replace(
                            "XXXXX", str(parameter_values[no_p]), 1
                        )
                        no_p += 1
                    content[idx] = line_temp
            f = open("frescox_temp_input.in", "a")
            f.writelines(content)
            f.close()

        def function(self):
            output_file = "48Ca_temp.out"
            input_file = "frescox_temp_input.in"
            os.system("frescox < frescox_temp_input.in > 48Ca_temp.out")
            
            # Read outputs
            with open(output_file) as f:
                content = f.readlines()
            cross_section = []
            for idline, line in enumerate(content):
                if "X-S" in line:
                    cross_section.append(float(line.split()[4]))
            os.remove(input_file)
            os.remove(output_file)
            for fname in os.listdir():
                if fname.startswith("fort"):
                    os.remove(fname)
            f = np.log(np.array(cross_section))
            f = f[
                np.array(
                    [[26, 31, 41, 51, 61, 71, 76, 81, 91, 101, 111, 121, 131, 141, 151]]
                )
            ]
            return f

        def sim(self, H, persis_info, sim_specs, libE_info):
            """
            Wraps frescox function
            """
            function = sim_specs["user"]["function"]
            H_o = np.zeros(1, dtype=sim_specs["out"])

            V = H["thetas"][0][0]
            r = H["thetas"][0][1]
            Ws = H["thetas"][0][2]

            # V = 49.2849
            # r = 0.9070
            a = 0.6798
            # Ws = 3.3944
            rs = 1.0941
            a2 = 0.2763

            parameter = [V, r, a, Ws, rs, a2]
            self.generate_input_file(parameter)
            H_o["f"] = function()
            for fname in os.listdir():
                if fname.startswith("fort"):
                    os.remove(fname)
            return H_o, persis_info

    design_start = time.time()

    args = parse_arguments()
    cls_fresco = bfrescox()
    print("Generating test data")
    test_data = generate_test_data(cls_fresco)
    print("End of test data generation")
    
    # Set a uniform prior
    prior_func = prior_dist(dist="uniform")(
        a=cls_fresco.thetalimits[:, 0], b=cls_fresco.thetalimits[:, 1]
    )

    print("Beginning of sequential procedure")
    al_fresco = designer(
        data_cls=cls_fresco,
        method="SEQCAL",
        args={
            "mini_batch": 1,
            "n_init_thetas": args.n_init_thetas,
            "nworkers": 2,
            "AL": args.al_func,
            "seed_n0": args.seed_n0,
            "prior": prior_func,
            "data_test": test_data,
            "max_evals": args.max_eval,
            "type_init": None,
        },
    )

    print("End of sequential procedure")
    theta_al = al_fresco._info["theta"]

    real_x = np.array(
        [[26, 31, 41, 51, 61, 71, 76, 81, 91, 101, 111, 121, 131, 141, 151]]
    ).T
    real_data = np.array(
        [
            [
                1243,
                887.7,
                355.5,
                111.5,
                26.5,
                10.4,
                8.3,
                7.3,
                17.2,
                37.6,
                48.7,
                38.9,
                32.4,
                36.4,
                61.9,
            ]
        ],
        dtype="float64",
    )
    n0 = 32
    f = al_fresco._info["f"]
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    plt.rcParams['font.family'] = 'DejaVu Sans' 
    for i in range(theta_al.shape[0]):
        if i < n0:
            if i > 0:
                ax.plot(
                    np.arange(15), np.exp(f[i, :]), color="gray", alpha=1, zorder=1
                )
            else:
                ax.plot(
                    np.arange(15),
                    np.exp(f[i, :]),
                    color="gray",
                    alpha=1,
                    zorder=1,
                    label="Initial sample",
                )
        else:
            if i < 63:
                ax.plot(
                    np.arange(15), np.exp(f[i, :]), color="red", alpha=0.3, zorder=1
                )
            else:
                ax.plot(
                    np.arange(15),
                    np.exp(f[i, :]),
                    color="red",
                    alpha=0.3,
                    zorder=1,
                    label="Acquired sample",
                )
    ax.scatter(
        np.arange(15), real_data.T, color="black", marker="P", zorder=2, label="Data"
    )
    ax.set_xticks(ticks=np.arange(15)[::3], labels=real_x.flatten()[::3])
    #ax.set_yticks(fontsize=16)
    ax.set_xlabel("Degree", fontsize=16)
    ax.set_ylabel("Cross section", fontsize=16)
    ax.set_yscale("log")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)
    plt.savefig("Figure_fresco.jpg", format="jpeg", bbox_inches="tight", dpi=500)
    plt.show()

    design_end = time.time()
    print("Elapsed time: " + str(round(design_end - design_start, 2)))
