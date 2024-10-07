import numpy as np
from PUQ.design import designer
from PUQ.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import create_test_non, add_result, samplingdata, observe_results
from ptest_funcs import pritam

args = parse_arguments()


ninit = 30
nmax = 180
result = []

args.seedmin = 0
args.seedmax = 1

if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):

        s = int(s)

        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        xr = np.array([[xx, yy] for xx in x for yy in y])
        xr = np.concatenate((xr, xr))
        cls_data = pritam()
        cls_data.realdata(xr, seed=s)

        prior_xt = prior_dist(dist="uniform")(
            a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]
        )
        prior_x = prior_dist(dist="uniform")(
            a=cls_data.thetalimits[0:2, 0], b=cls_data.thetalimits[0:2, 1]
        )
        prior_t = prior_dist(dist="uniform")(
            a=np.array([cls_data.thetalimits[2][0]]),
            b=np.array([cls_data.thetalimits[2][1]]),
        )

        priors = {"prior": prior_xt, "priorx": prior_x, "priort": prior_t}

        xt_test, ftest, ptest, thetamesh, xmesh = create_test_non(cls_data)
        ytest = cls_data.function(
            xmesh[:, 0], xmesh[:, 1], cls_data.true_theta
        ).reshape(1, len(xmesh))

        test_data = {
            "theta": xt_test,
            "f": ftest,
            "p": ptest,
            "y": ytest,
            "th": thetamesh,
            "xmesh": xmesh,
            "p_prior": 1,
        }

        al_ceivarx = designer(
            data_cls=cls_data,
            method="SEQDES",
            args={
                "mini_batch": 1,
                "n_init_thetas": ninit,
                "nworkers": 2,
                "AL": "ceivarx",
                "seed_n0": s,
                "prior": priors,
                "data_test": test_data,
                "max_evals": nmax,
                "theta_torun": None,
                "is_thetamle": False,
            },
        )

        xt_eivarx = al_ceivarx._info["theta"]
        f_eivarx = al_ceivarx._info["f"]

        save_output(al_ceivarx, cls_data.data_name, "ceivarx", 2, 1, s)

        res = {
            "method": "eivarx",
            "repno": s,
            "Prediction Error": al_ceivarx._info["TV"],
            "Posterior Error": al_ceivarx._info["HD"],
        }
        result.append(res)

        # # # # # # # # # # # # # # # # # # # # #
        al_ceivar = designer(
            data_cls=cls_data,
            method="SEQDES",
            args={
                "mini_batch": 1,
                "n_init_thetas": ninit,
                "nworkers": 2,
                "AL": "ceivar",
                "seed_n0": s,
                "prior": priors,
                "data_test": test_data,
                "max_evals": nmax,
                "theta_torun": None,
                "is_thetamle": False,
            },
        )

        xt_eivar = al_ceivar._info["theta"]
        f_eivar = al_ceivar._info["f"]

        save_output(al_ceivar, cls_data.data_name, "ceivar", 2, 1, s)

        res = {
            "method": "eivar",
            "repno": s,
            "Prediction Error": al_ceivar._info["TV"],
            "Posterior Error": al_ceivar._info["HD"],
        }
        result.append(res)

        # LHS
        xt_LHS = samplingdata("LHS", nmax - ninit, cls_data, s, prior_xt)
        al_LHS = designer(
            data_cls=cls_data,
            method="SEQDES",
            args={
                "mini_batch": 1,
                "n_init_thetas": ninit,
                "nworkers": 2,
                "AL": None,
                "seed_n0": s,
                "prior": priors,
                "data_test": test_data,
                "max_evals": nmax,
                "theta_torun": xt_LHS,
                "is_thetamle": False,
            },
        )
        xt_LHS = al_LHS._info["theta"]
        f_LHS = al_LHS._info["f"]

        save_output(al_LHS, cls_data.data_name, "lhs", 2, 1, s)

        res = {
            "method": "lhs",
            "repno": s,
            "Prediction Error": al_LHS._info["TV"],
            "Posterior Error": al_LHS._info["HD"],
        }
        result.append(res)

        # rnd
        xt_RND = samplingdata("Random", nmax - ninit, cls_data, s, prior_xt)
        al_RND = designer(
            data_cls=cls_data,
            method="SEQDES",
            args={
                "mini_batch": 1,
                "n_init_thetas": ninit,
                "nworkers": 2,
                "AL": None,
                "seed_n0": s,
                "prior": priors,
                "data_test": test_data,
                "max_evals": nmax,
                "theta_torun": xt_RND,
                "is_thetamle": False,
            },
        )
        xt_RND = al_RND._info["theta"]
        f_RND = al_RND._info["f"]

        save_output(al_RND, cls_data.data_name, "rnd", 2, 1, s)

        res = {
            "method": "rnd",
            "repno": s,
            "Prediction Error": al_RND._info["TV"],
            "Posterior Error": al_RND._info["HD"],
        }
        result.append(res)

    method = ["eivarx", "eivar", "lhs", "rnd"]
    observe_results(result, method, args.seedmax - args.seedmin, ninit, nmax)
