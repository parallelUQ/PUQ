import numpy as np
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    get_emuvar,
    multiple_pdfs,
)
from PUQ.designmethods.gen_funcs.EIVAR import eivar
from PUQ.designmethods.gen_funcs.MAXEXP import maxexp
from PUQ.designmethods.gen_funcs.MAXVAR import maxvar
from PUQ.designmethods.gen_funcs.EI import ei
from PUQ.designmethods.gen_funcs.PI import pi
from PUQ.designmethods.gen_funcs.HYBRID_EI import hybrid_ei
from PUQ.designmethods.gen_funcs.RND import rnd
from PUQ.designmethods.SEQCALsupport import (
    fit_emulator,
    load_H,
    update_arrays,
    create_arrays,
    pad_arrays,
    rebuild_condition_opt,
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
from smt.sampling_methods import LHS
from PUQ.posterior import posterior
import time


def fit(fitinfo, data_cls, args):

    mini_batch = args["mini_batch"]
    nworkers = args["nworkers"]
    AL_type = args["AL"]
    seed_n0 = args["seed_n0"]
    prior = args["prior"]
    max_evals = args["max_evals"]
    test_data = args["data_test"]

    if AL_type in ["ei", "pi", "hybrid_ei", "hybrid_pi", "eivar"]:
        candsize = args["candsize"]
        refsize = args["refsize"]
        believer = args["believer"]
    else:
        candsize, refsize, believer = None, None, None

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
        ("AE", float),
        ("time", float),
    ]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": [o[0] for o in gen_out] + ["f", "sim_id"],
        "out": gen_out,
        "user": {
            "mini_batch": mini_batch,  # No. of thetas to generate per step
            "nworkers": nworkers,
            "AL": AL_type,
            "seed_n0": seed_n0,
            "synth_cls": data_cls,
            "test_data": test_data,
            "prior": prior,
            "candsize": candsize,
            "refsize": refsize,
            "believer": believer,
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
    # libE_specs = {'nworkers': nworkers, 'comms': 'local', 'sim_dirs_make': True,
    #              'sim_dir_copy_files': [os.path.join(os.getcwd(), '48Ca_template.in')]}

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

    emu = fit_emulator(data_cls.x, H["thetas"], H["f"][None, :], data_cls.thetalimits)

    TV, HD, AE = collect_data(
        emu,
        data_cls.x,
        H["f"][None, :],
        data_cls.x,
        test_data["theta"],
        test_data["p_prior"],
        test_data["p"],
        np.reshape(data_cls.real_data[0, :], (1, data_cls.real_data.shape[1])),
        data_cls.obsvar.reshape(1, data_cls.d, data_cls.d),
    )

    fitinfo["f"] = H["f"]
    fitinfo["theta"] = H["thetas"]
    fitinfo["TV"] = np.concatenate((H["TV"], np.array([TV])))
    fitinfo["HD"] = np.concatenate((H["HD"], np.array([HD])))
    fitinfo["AE"] = np.concatenate((H["AE"], np.array([AE])))

    fitinfo["time"] = H["time"]
    return


def compute_timepass(start_time, new_theta):
    end_time = time.time()
    timepass = end_time - start_time
    timepassvec = np.zeros(new_theta.shape[0])
    timepassvec[0] = timepass
    return timepassvec


def collect_data(
    emu, x, fevals, real_x, thetatest, priortest, posttest, true_fevals, obsvar3d
):

    emupredict = emu.predict(x=x, theta=thetatest)
    emumean = emupredict.mean()
    emuvar, is_cov = get_emuvar(emupredict)
    emumeanT = emumean.T
    emuvarT = emuvar.transpose(1, 0, 2)
    var_obsvar1 = emuvarT + obsvar3d

    posttesthat = multiple_pdfs(
        true_fevals, emumeanT[:, real_x.flatten()], var_obsvar1[:, real_x, real_x.T]
    )

    TV = np.mean(np.abs(posttest - posttesthat * priortest))
    HD = np.sqrt(0.5 * np.mean((np.sqrt(posttesthat) - np.sqrt(posttest)) ** 2))
    idnan = np.isnan(fevals).any(axis=0).flatten()
    fevals_c = fevals[:, ~idnan]
    AE = np.min(np.sum(np.abs(true_fevals - fevals_c.T), axis=1))

    return TV, HD, AE


def gen_f(H, persis_info, gen_specs, libE_info):
    """Generator to select and obviate parameters for calibration."""
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    rand_stream = persis_info["rand_stream"]
    mini_batch = gen_specs["user"]["mini_batch"]
    n_workers = gen_specs["user"]["nworkers"]
    AL = gen_specs["user"]["AL"]
    seed = gen_specs["user"]["seed_n0"]
    synth_info = gen_specs["user"]["synth_cls"]
    test_data = gen_specs["user"]["test_data"]
    prior_func = gen_specs["user"]["prior"]
    candsize = gen_specs["user"]["candsize"]
    refsize = gen_specs["user"]["refsize"]
    believer = gen_specs["user"]["believer"]

    obsvar = synth_info.obsvar
    data = synth_info.real_data
    theta_limits = synth_info.thetalimits

    thetatest, posttest, ftest, priortest = None, None, None, None
    if test_data is not None:
        thetatest, posttest, ftest, priortest, thetainit, finit = (
            test_data["theta"],
            test_data["p"],
            test_data["f"],
            test_data["p_prior"],
            test_data["thetainit"],
            test_data["finit"],
        )

    true_fevals = np.reshape(data[0, :], (1, data.shape[1]))
    n_x, x, real_x = synth_info.d, synth_info.x, synth_info.real_x
    obsvar3d = obsvar.reshape(1, n_x, n_x)
    obs_offset, theta_offset, generated_no = 0, 0, 0
    TV, HD, AE, time_pass = 1000, 1000, 1000, 0
    fevals, pending, prev_pending, complete, prev_complete = (
        None,
        None,
        None,
        None,
        None,
    )
    first_iter = True
    tag = 0
    update_model = False
    acquisition_f = eval(AL)
    list_id = []
    theta = 0

    while tag not in [STOP_TAG, PERSIS_STOP]:
        starttime = time.time()
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
            update_model = rebuild_condition_opt(
                complete, prev_complete, n_theta=mini_batch
            )

            if not update_model:
                tag, Work, calc_in = ps.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break

        if update_model:
            starttime = time.time()

            # if len(theta) % 50 == 0:
            print("Updating model...\n")

            print(
                "Percentage Pending: %0.2f ( %d / %d)"
                % (
                    100 * np.round(np.mean(pending), 4),
                    np.sum(pending),
                    np.prod(pending.shape),
                )
            )
            print(
                "Percentage Complete: %0.2f ( %d / %d)"
                % (
                    100 * np.round(np.mean(complete), 4),
                    np.sum(complete),
                    np.prod(pending.shape),
                )
            )

            fcomb = np.concatenate((finit, fevals), axis=1)
            thetacomb = np.concatenate((thetainit, theta), axis=0)
            emu = fit_emulator(x, thetacomb, fcomb, theta_limits)
            prev_pending = pending.copy()
            update_model = False

            # Obtain the accuracy on the test set
            if test_data is not None:
                TV, HD, AE = collect_data(
                    emu,
                    x,
                    fcomb,
                    real_x,
                    thetatest,
                    priortest,
                    posttest,
                    true_fevals,
                    obsvar3d,
                )

        if first_iter:
            emuinit = fit_emulator(x, thetainit, finit, theta_limits)
            TV, HD, AE = collect_data(
                emuinit,
                x,
                finit,
                real_x,
                thetatest,
                priortest,
                posttest,
                true_fevals,
                obsvar3d,
            )
            n_init = n_workers - 1

            theta = acquisition_f(
                n_init,
                x,
                real_x,
                emuinit,
                thetainit,
                finit,
                true_fevals,
                obsvar,
                theta_limits,
                prior_func,
                thetatest,
                priortest,
                None,
                believer=believer,
                candsize=candsize,
                refsize=refsize,
            )

            fevals, pending, prev_pending, complete, prev_complete = create_arrays(
                n_x, n_init
            )
            time_pass = compute_timepass(starttime, theta)
            H_o = np.zeros(len(theta), dtype=gen_specs["out"])
            H_o = load_H(
                H_o, theta, TV, HD, AE, time_pass, generated_no, set_priorities=True
            )
            tag, Work, calc_in = ps.send_recv(H_o)
            first_iter = False
            generated_no += n_init

        else:
            if rebuild_condition_opt(complete, prev_complete, n_theta=mini_batch):

                prev_complete = complete.copy()
                new_theta = acquisition_f(
                    mini_batch,
                    x,
                    real_x,
                    emu,
                    thetacomb,
                    fcomb,
                    true_fevals,
                    obsvar,
                    theta_limits,
                    prior_func,
                    thetatest,
                    priortest,
                    None,
                    believer=believer,
                    candsize=candsize,
                    refsize=refsize,
                )

                theta, fevals, pending, prev_pending, complete, prev_complete = (
                    pad_arrays(
                        n_x,
                        new_theta,
                        theta,
                        fevals,
                        pending,
                        prev_pending,
                        complete,
                        prev_complete,
                    )
                )

                time_pass = compute_timepass(starttime, new_theta)

                H_o = np.zeros(len(new_theta), dtype=gen_specs["out"])
                H_o = load_H(
                    H_o,
                    new_theta,
                    TV,
                    HD,
                    AE,
                    time_pass,
                    generated_no,
                    set_priorities=True,
                )
                tag, Work, calc_in = ps.send_recv(H_o)
                generated_no += mini_batch

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
