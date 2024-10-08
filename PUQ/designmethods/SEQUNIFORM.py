import numpy as np
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    get_emuvar,
    multiple_pdfs,
)
from PUQ.designmethods.gen_funcs.acquisition_funcs import maxvar, eivar, maxexp, rnd
from PUQ.designmethods.SEQCALsupport import (
    fit_emulator,
    load_H,
    update_arrays,
    create_arrays,
    pad_arrays,
    select_condition,
    rebuild_condition,
)
from libensemble.message_numbers import (
    STOP_TAG,
    PERSIS_STOP,
    FINISHED_PERSISTENT_GEN_TAG,
    EVAL_GEN_TAG,
)
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.alloc_funcs.start_only_persistent import (
    only_persistent_gens as alloc_f,
)
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from PUQ.prior import prior_dist


def fit(fitinfo, data_cls, args):
    mini_batch = args["mini_batch"]
    n_init_thetas = args["n_init_thetas"]
    nworkers = args["nworkers"]
    max_evals = args["max_evals"]

    out = data_cls.out
    sim_f = data_cls.sim

    sim_specs = {
        "sim_f": sim_f,
        "in": ["thetas"],
        "out": out,
        "user": {"function": data_cls.function},
    }

    gen_out = [
        ("thetas", float, data_cls.p),
        ("priority", int),
        ("obs", float, (1,)),
        ("obsvar", float, (1,)),
        ("TV", float),
        ("HD", float),
    ]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": [o[0] for o in gen_out] + ["f", "sim_id"],
        "out": gen_out,
        "user": {
            "n_init_thetas": n_init_thetas,  # Num thetas in initial batch
            "mini_batch": mini_batch,  # No. of thetas to generate per step
            "nworkers": nworkers,
            "synth_cls": data_cls,
        },
    }

    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "init_sample_size": 0,
            "async_return": True,  # True = Return results to gen as they come in (after sample)
            "active_recv_gen": True,  # Persistent gen can handle irregular communications
        },
    }
    libE_specs = {"nworkers": nworkers, "comms": "local"}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Currently just allow gen to exit if mse goes below threshold value
    exit_criteria = {"sim_max": max_evals}  # Now just a set number of sims.

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs,
        gen_specs,
        exit_criteria,
        persis_info,
        alloc_specs=alloc_specs,
        libE_specs=libE_specs,
    )

    fitinfo["f"] = H["f"]
    fitinfo["theta"] = H["thetas"]
    fitinfo["TV"] = H["TV"]
    fitinfo["HD"] = H["HD"]
    return


def gen_f(H, persis_info, gen_specs, libE_info):
    """Generator to select and obviate parameters for calibration."""
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    rand_stream = persis_info["rand_stream"]
    n0 = gen_specs["user"]["n_init_thetas"]
    mini_batch = gen_specs["user"]["mini_batch"]
    n_workers = gen_specs["user"]["nworkers"]
    synth_info = gen_specs["user"]["synth_cls"]

    theta_torun = synth_info.theta
    obsvar = synth_info.obsvar
    data = synth_info.real_data
    theta_limits = synth_info.thetalimits

    true_fevals = np.reshape(data[0, :], (1, data.shape[1]))

    n_x = synth_info.d
    n_realx = true_fevals.shape[1]
    x = synth_info.x
    real_x = synth_info.real_x

    obs_offset, theta_offset, generated_no = 0, 0, 0
    TV, HD = 1000, 1000
    fevals, pending, prev_pending, complete, prev_complete = (
        None,
        None,
        None,
        None,
        None,
    )
    first_iter = True
    tag = 0

    obsvar3d = obsvar.reshape(1, n_x, n_x)
    update_model = False
    list_id = []

    theta = 0

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if not first_iter:
            # Update fevals from calc_in
            update_arrays(
                n_x,
                fevals,
                pending,
                complete,
                calc_in,
                obs_offset,
                theta_offset,
                list_id,
            )
            update_model = rebuild_condition(
                complete, prev_complete, n_theta=mini_batch, n_initial=n0
            )

            if not update_model:
                tag, Work, calc_in = ps.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break

        if first_iter:
            # print('Selecting theta for the first iteration...\n')

            n_init = max(n_workers - 1, n0)
            theta = theta_torun[0:n_init, :]
            fevals, pending, prev_pending, complete, prev_complete = create_arrays(
                n_x, n_init
            )

            H_o = np.zeros(len(theta), dtype=gen_specs["out"])
            H_o = load_H(H_o, theta, TV, HD, generated_no, set_priorities=True)
            tag, Work, calc_in = ps.send_recv(H_o)
            first_iter = False
            generated_no += n_init

        else:
            if select_condition(
                complete, prev_complete, n_theta=mini_batch, n_initial=n0
            ):
                # print('Selecting theta...\n')

                prev_complete = complete.copy()
                new_theta = theta_torun[generated_no : (generated_no + mini_batch), :]

                (
                    theta,
                    fevals,
                    pending,
                    prev_pending,
                    complete,
                    prev_complete,
                ) = pad_arrays(
                    n_x,
                    new_theta,
                    theta,
                    fevals,
                    pending,
                    prev_pending,
                    complete,
                    prev_complete,
                )

                H_o = np.zeros(len(new_theta), dtype=gen_specs["out"])
                H_o = load_H(H_o, new_theta, TV, HD, generated_no, set_priorities=True)
                tag, Work, calc_in = ps.send_recv(H_o)
                generated_no += mini_batch

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
