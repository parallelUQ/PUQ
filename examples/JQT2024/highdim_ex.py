import numpy as np
from PUQ.design import designer
from PUQ.utils import parse_arguments, save_output
from PUQ.prior import prior_dist
from plots_design import create_test_highdim, add_result, samplingdata
from ptest_funcs import highdim2

args = parse_arguments()

ninit = 50
nmax = 200
result = []

size_x = 2

if __name__ == "__main__":
    for s in np.arange(args.seedmin, args.seedmax):
        s = int(s)
        xr = np.concatenate(
            (
                np.repeat(0.5, size_x)[None, :],
                np.repeat(0.5, size_x)[None, :],
                np.repeat(0.5, size_x)[None, :],
                np.repeat(0.5, size_x)[None, :],
            ),
            axis=0,
        )
        # xr = np.concatenate((np.repeat(0.5, size_x)[None, :], np.repeat(0.5, size_x)[None, :], np.repeat(0.5, size_x)[None, :]), axis=0)
        # xr = np.concatenate((np.repeat(0.5, size_x)[None, :], np.repeat(0.5, size_x)[None, :]), axis=0)

        cls_data = highdim2()
        cls_data.realdata(xr, seed=s)
        print(cls_data.sigma2)

        prior_xt = prior_dist(dist="uniform")(
            a=cls_data.thetalimits[:, 0], b=cls_data.thetalimits[:, 1]
        )
        prior_x = prior_dist(dist="uniform")(
            a=cls_data.thetalimits[0:size_x, 0], b=cls_data.thetalimits[0:size_x, 1]
        )
        prior_t = prior_dist(dist="uniform")(
            a=cls_data.thetalimits[size_x:, 0], b=cls_data.thetalimits[size_x:, 1]
        )

        priors = {"prior": prior_xt, "priorx": prior_x, "priort": prior_t}

        xt_test, ftest, ptest, thetamesh, xmesh = create_test_highdim(cls_data)

        if size_x == 2:
            ytest = cls_data.function(
                xmesh[:, 0],
                xmesh[:, 1],
                cls_data.true_theta[0],
                cls_data.true_theta[1],
                cls_data.true_theta[2],
                cls_data.true_theta[3],
                cls_data.true_theta[4],
                cls_data.true_theta[5],
                cls_data.true_theta[6],
                cls_data.true_theta[7],
                cls_data.true_theta[8],
                cls_data.true_theta[9],
            ).reshape(1, len(xmesh))
        elif size_x == 6:
            ytest = cls_data.function(
                xmesh[:, 0],
                xmesh[:, 1],
                xmesh[:, 2],
                xmesh[:, 3],
                xmesh[:, 4],
                xmesh[:, 5],
                cls_data.true_theta[0],
                cls_data.true_theta[1],
                cls_data.true_theta[2],
                cls_data.true_theta[3],
                cls_data.true_theta[4],
                cls_data.true_theta[5],
            ).reshape(1, len(xmesh))
        elif size_x == 10:
            ytest = cls_data.function(
                xmesh[:, 0],
                xmesh[:, 1],
                xmesh[:, 2],
                xmesh[:, 3],
                xmesh[:, 4],
                xmesh[:, 5],
                xmesh[:, 6],
                xmesh[:, 7],
                xmesh[:, 8],
                xmesh[:, 9],
                cls_data.true_theta[0],
                cls_data.true_theta[1],
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

        al_imspe = designer(
            data_cls=cls_data,
            method="SEQDES",
            args={
                "mini_batch": 1,
                "n_init_thetas": ninit,
                "nworkers": 2,
                "AL": "imspe",
                "seed_n0": s,
                "prior": priors,
                "data_test": test_data,
                "max_evals": nmax,
                "theta_torun": None,
                "is_thetamle": False,
            },
        )

        save_output(al_imspe, cls_data.data_name, "imspe", 2, 1, s)

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

        save_output(al_ceivar, cls_data.data_name, "ceivar", 2, 1, s)

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

        save_output(al_ceivarx, cls_data.data_name, "ceivarx", 2, 1, s)

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

        save_output(al_LHS, cls_data.data_name, "lhs", 2, 1, s)

        al_maxvar = designer(
            data_cls=cls_data,
            method="SEQDES",
            args={
                "mini_batch": 1,
                "n_init_thetas": ninit,
                "nworkers": 2,
                "AL": "maxvar",
                "seed_n0": s,
                "prior": priors,
                "data_test": test_data,
                "max_evals": nmax,
                "theta_torun": None,
                "is_thetamle": False,
            },
        )

        save_output(al_maxvar, cls_data.data_name, "maxvar", 2, 1, s)
