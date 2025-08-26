import numpy as np
from smt.sampling_methods import LHS
from PUQ.designmethods.support import multiple_pdfs, multiple_determinants
from hetgpy.covariance_functions import cov_gen
from hetgpy.IMSE import crit_IMSPE, Wij, IMSPE, allocate_mult
import emcee
from PUQ.designmethods.gen_funcs.allocate_reps import allocate


def generate_neighborhood(acq):
    
    rand = np.random.default_rng(acq.seed)
    neigh_type = acq.args.get("neighbor", None)
    
    if neigh_type == "LHS":
        N = int(acq.nL)
        sampling = LHS(xlimits=acq.zlim, random_state=int(acq.seed))
        L = sampling(N)
    else:
        N = int(acq.nL * 0.5)

        sampling = LHS(xlimits=acq.zlim, random_state=int(acq.seed))
        L_explore = sampling(N)
   
        sampling = LHS(xlimits=acq.tlim, random_state=int(acq.seed))
        Lt = sampling(N)

        num_options = len(acq.x)
    
        # Distribute the rows as evenly as possible
        counts = np.full(num_options, N // num_options)  # Base count for each row type
        counts[
            rand.choice(num_options, N % num_options, replace=False)
        ] += 1  # Assign extra rows randomly
    
        # Create the array by stacking the chosen rows (Fixed: Explicitly convert to a list)
        Lx = np.vstack(
            [row for i in range(num_options) for row in [acq.x[i]] * counts[i]]
        )
    
        # Shuffle the rows randomly
        rand.shuffle(Lx)
        
        L_exploit = np.concatenate((Lx, Lt), axis=1)
    
        L = np.concatenate((L_explore, L_exploit))
    
    return L


class acquisition_function:
    def __init__(self, model, cls_func, args):
        self.model = model.copy()
        self.cls_func = cls_func
        #self.persis_info = persis_info
        self.args = args
        self.x = self.cls_func.x
        self.d = self.cls_func.d
        self.p = self.cls_func.p
        self.dx = self.cls_func.dx
        self.dt = self.cls_func.dt
        self.zlim = cls_func.zlim
        self.xlim = cls_func.zlim[0 : cls_func.dx, :]
        self.tlim = cls_func.zlim[cls_func.dx : cls_func.p, :]
        self.y = self.cls_func.real_data
        self.Sigma = self.cls_func.obsvar
        self.Sigma3d = self.Sigma.reshape(1, self.d, self.d)
        self.detSigma = np.linalg.det(self.Sigma)
        self.twopid = (2**self.d) * (np.sqrt(np.pi) ** self.d)
        self.twopiddet = (
            (2**self.d) * (np.sqrt(np.pi) ** self.d) * np.sqrt(self.detSigma)
        )

    def acquire_new(self):

        if isinstance(self, lookahead):
            return eval("self.evaluate")()

        else:
            return eval("self.evaluate")(
                self.args.get("new", True), return_pseudo=False
            )

    def gen_pred(self, model, x, t, return_id=False, return_flat=False):
        nm, d = t.shape[0], x.shape[0]
        ntot = nm * d

        # print(t.shape)
        # (ntot, d)
        x_tiled = np.tile(x, (t.shape[0], 1))
        # (ntot, p-d)
        t_repeated = np.repeat(t, x.shape[0], axis=0)
        # (ntot, p)
        z = np.hstack([x_tiled, t_repeated])

        # to construct S matrix
        id_row = np.arange(0, ntot)
        id_col = np.arange(0, ntot).reshape(nm, d)
        id_col = np.repeat(id_col, repeats=d, axis=0)

        # predict at mesh
        meshPr = model.predict(x=z, xprime=z)

        # ntot, ntot x ntot, ntot
        mu, Sn, sd2 = meshPr["mean"], meshPr["cov"], meshPr["sd2"]

        muT = mu.reshape(nm, d)
        S = Sn[id_row[:, None], id_col].reshape(nm, d, d)

        if return_id:
            return muT, S, z, id_row[:, None], id_col, mu[:, None]
        else:
            if return_flat:
                return muT, S, z, mu[:, None]
            else:
                return muT, S, z

    def crit_pvar(self, model, x, t):

        # posterior variance
        mu, S, z = self.gen_pred(model, x, t)

        M = S + 0.5 * self.Sigma3d
        N = S + self.Sigma3d

        f = multiple_pdfs(self.y, mu, M)
        g = multiple_pdfs(self.y, mu, N)

        vals = (1 / self.twopiddet) * f - g**2

        return vals

    def total_var(self, model, x, t):
        vals = self.crit_pvar(model=model, x=x, t=t)
        return np.mean(self.weights * vals)


class var(acquisition_function):
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.explore = args.get("explore", True)
        self.nL = args.get("nL", 100)
        
        if args.get("integral") == "importance":
            # print("Importance sampling")
            self.epsilon = args.get("epsilon", 10 ** (-10))
            self.discard = args.get("discard", 100)
            self.nsteps = args.get("nsteps", 200)
            self.thin = args.get("thin", 20)
            self.nwalkers = args.get("nwalkers", 20)
            self.reference_set()
        elif args.get("integral") == "LHS":
            print("LHS")
            sampling = LHS(xlimits=self.tlim, random_state=int(self.seed))
            self.t_ref = sampling(500)
            self.weights = 1
        else:
            self.t_ref = args.get("t_grid")
            self.weights = 1
            
    def reference_set(self):
        def log_probability(ctheta):
            if np.any((ctheta < 0) | (ctheta > 1)):
                return -np.inf
            else:
                pvar = self.crit_pvar(model=self.model, x=self.x, t=ctheta[None, :])

                pvar = max(pvar, 0)
                return np.log(pvar)

        def sample(ndim, nwalkers):
            np.random.seed(int(self.seed))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

            sampling = LHS(xlimits=self.tlim, random_state=int(self.seed))
            loc0 = sampling(nwalkers)

            sampler.run_mcmc(initial_state=loc0, nsteps=self.nsteps, progress=False)

            samples = sampler.get_chain(discard=self.discard, thin=self.thin, flat=True)
            return samples

        def importance_weight(theta):
            pvar = self.crit_pvar(model=self.model, x=self.x, t=theta)
            pvar = pvar + self.epsilon
            unnorm_weight = 1 / pvar
            weight = unnorm_weight / np.sum(unnorm_weight)
            return weight

        self.t_ref = sample(ndim=self.dt, nwalkers=self.nwalkers)
        self.weights = importance_weight(theta=self.t_ref)

    def evaluate(self, new, return_pseudo):

        if new:
            L = generate_neighborhood(self)
            new_input = self.evaluate_explore(L)
        else:
            L = self.model["X0"]
            new_input = self.evaluate_exploit(L)

        self.znew, self.new = new_input, new

        if return_pseudo:
            pnew = self.model.predict(x=self.znew)
            fnew = np.array([pnew["mean"]])[None, :]
            self.fnew = fnew

        return self

    def evaluate_explore(self, L):
        vals = self.crit_pvar(model=self.model, x=self.x, t=L[:, self.dx : self.p])

        new_input = L[np.argmax(vals), :].reshape(1, self.p)
        return new_input

    def evaluate_exploit(self, L):
        vals = self.crit_pvar(model=self.model, x=self.x, t=L[:, self.dx : self.p])

        new_input = L[np.argmax(vals), :].reshape(1, self.p)
        return new_input


class imse(acquisition_function):
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.explore = args.get("explore", "both")
        self.nL = args.get("nL", 100)
        self.t_grid = args.get("t_grid", None)

    def evaluate(self, new, return_pseudo):

        if new:
            sampling = LHS(xlimits=self.zlim, random_state=int(self.seed))
            L = sampling(self.nL)
            new_input = self.evaluate_explore(L)
        else:
            L = self.model["X0"]
            new_input = self.evaluate_exploit(L)

        self.znew, self.new = new_input, new

        if return_pseudo:
            pnew = self.model.predict(x=self.znew)
            fnew = np.array([pnew["mean"]])[None, :]
            self.fnew = fnew

        return self

    def evaluate_explore(self, L):
        IMSPE_grid = np.array([crit_IMSPE(x, model=self.model) for x in L])
        new_input = L[np.argmin(IMSPE_grid), :].reshape(1, self.cls_func.p)
        return new_input

    def evaluate_exploit(self, L):
        IMSPE_grid = np.array(
            [crit_IMSPE(id=[x_id], model=self.model) for x_id, x in enumerate(L)]
        )
        new_input = L[np.argmin(IMSPE_grid), :].reshape(1, self.cls_func.p)
        return new_input


class ivar(acquisition_function):
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.explore = args.get("explore", "both")
        self.nL = args.get("nL", 100)
        if args.get("integral") == "importance":
            # print("Importance sampling")
            self.epsilon = args.get("epsilon", 10 ** (-10))
            self.discard = args.get("discard", 100)
            self.nsteps = args.get("nsteps", 200)
            self.thin = args.get("thin", 20)
            self.nwalkers = args.get("nwalkers", 20)
            self.reference_set()
        elif args.get("integral") == "LHS":
            print("LHS")
            sampling = LHS(xlimits=self.tlim, random_state=int(self.seed))
            self.t_ref = sampling(500)
            self.weights = 1
        else:
            self.t_ref = args.get("t_grid")
            self.weights = 1

    def reference_set(self):
        def log_probability(ctheta):
            if np.any((ctheta < 0) | (ctheta > 1)):
                return -np.inf
            else:
                pvar = self.crit_pvar(model=self.model, x=self.x, t=ctheta[None, :])

                pvar = max(pvar, 0)
                return np.log(pvar)

        def sample(ndim, nwalkers):
            np.random.seed(int(self.seed))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

            sampling = LHS(xlimits=self.tlim, random_state=int(self.seed))
            loc0 = sampling(nwalkers)

            sampler.run_mcmc(initial_state=loc0, nsteps=self.nsteps, progress=False)

            samples = sampler.get_chain(discard=self.discard, thin=self.thin, flat=True)
            return samples

        def importance_weight(theta):
            pvar = self.crit_pvar(model=self.model, x=self.x, t=theta)
            pvar = pvar + self.epsilon
            unnorm_weight = 1 / pvar
            weight = unnorm_weight / np.sum(unnorm_weight)
            return weight

        self.t_ref = sample(ndim=self.dt, nwalkers=self.nwalkers)

        # import pandas as pd
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # df = pd.DataFrame(self.t_ref)
        # sns.pairplot(df, diag_kind="kde")  # Use KDE for diagonal histograms
        # plt.show()
        # print(self.t_ref.shape)
        # import matplotlib.pyplot as plt
        # if self.t_ref.shape[1] > 1:
        #     plt.scatter(self.t_ref[:, 0], self.t_ref[:, 1])
        #     plt.show()
        # else:
        #     plt.hist(self.t_ref)
        #     plt.show()

        self.weights = importance_weight(theta=self.t_ref)

    def evaluate(self, new, return_pseudo):

        if new:
            L = generate_neighborhood(self)
            new_input = self.evaluate_explore(L)
        else:
            L = self.model["X0"]
            new_input = self.evaluate_exploit(L)

        self.znew, self.new = new_input, new

        if return_pseudo:
            pnew = self.model.predict(x=self.znew)
            fnew = np.array([pnew["mean"]])[None, :]
            self.fnew = fnew

        return self

    def evaluate_explore(self, L):


        nm = self.t_ref.shape[0]

        vals = np.zeros(self.nL)

        mu, S, z = self.gen_pred(self.model, self.x, self.t_ref)

        V1 = S + self.Sigma3d
        
        # for i in np.arange(S.shape[0]):
        #     print(np.diag(S[i, :, :]))
   
        # predict at candidates
        candPr = self.model.predict(x=L, xprime=z)

        # nL, nL, nL x ntot
        candvar, candnugs, candcov = candPr["sd2"], candPr["nugs"], candPr["cov"]
        candtotvat = candvar + candnugs

        for k in range(0, self.nL):
            # (mn x d x 1)
            covc = candcov[k, :].reshape(nm, self.d, 1)

            # (mn x 1 x d)
            covcT = np.transpose(covc, (0, 2, 1))

            # (nm, d, d)
            phic = (covc * covcT) / candtotvat[k]

            # C1: nm x d x d, C2: nm x d x d
            C1 = (V1 + phic) * 0.5
            C2 = V1 - phic

            rpdf = multiple_pdfs(self.y, mu, C1)
            dets = multiple_determinants(C2)

            part2 = (1 / self.twopid) * (rpdf / np.sqrt(dets))

            vals[k] = np.sum(self.weights * part2)

        new_input = L[np.argmax(vals), :].reshape(1, self.p)

        return new_input

    def evaluate_exploit(self, L):

        nL = L.shape[0]

        nm = self.t_ref.shape[0]
        vals = np.zeros(nL)
        nrep = 1

        mu, S, z, idr, idc, muf = self.gen_pred(
            self.model, self.x, self.t_ref, return_id=True
        )

        V1 = S + self.Sigma3d
        V2 = S + 0.5 * self.Sigma3d

        Ki, mult = self.model.Ki, self.model.mult

        candPr = self.model.predict(x=self.model["X0"])
        candmean, candvar, candnugs = candPr["mean"], candPr["sd2"], candPr["nugs"]
        smean = self.model["Z0"]

        # compute B
        if self.model.get("Lambda") is None:
            tmp = self.model.g
            tmp = np.repeat(tmp, nL)
        else:
            tmp = self.model.Lambda

        denom = (mult * (mult + nrep)) / (nrep * tmp) - np.diag(Ki)
        Ki3D = Ki[:, :, None] * Ki[:, None, :]
        B = Ki3D / denom[:, None, None]
        KB = self.model.Ki + B

        # ntot x ntr
        kx = cov_gen(
            X1=z, X2=self.model.X0, theta=self.model.theta, type=self.model.covtype
        )

        # coefficients to be used
        ap1 = mult + 1
        mean_coefs = (candmean - smean) / ap1
        var_coefs = (candvar + candnugs) / (ap1**2)
        smean = smean[:, None]

        KBe_all = KB[np.arange(nL), :, np.arange(nL)]
        kKBe_all = KBe_all @ kx.T

        B_kx = np.einsum("ijk, bk -> ijb", B, kx)
        phi_all = self.model.nu_hat * np.einsum("aj, ijb -> aib", kx, B_kx)

        Bs = B @ smean
        c1, c2 = (1 / self.twopiddet), (1 / self.twopid)
        for k in range(0, nL):

            phi_k = phi_all[idr, k, idc].reshape(nm, self.d, self.d)

            KBe_k = KB[k, :, k : k + 1]
            mu_k = muf + kx @ (mean_coefs[k] * KBe_k + Bs[k, :, :])
            mu_kT = mu_k.reshape(nm, self.d)

            kKBe = kKBe_all[k, :]
            covc = kKBe.reshape(nm, self.d, 1)
            covcT = np.transpose(covc, (0, 2, 1))

            # nm x d x d
            gamma_k = var_coefs[k] * (covc * covcT)

            M = V2 - phi_k + gamma_k
            part1 = c1 * multiple_pdfs(self.y, mu_kT, M)

            C1 = V1 - phi_k
            N = 0.5 * C1 + gamma_k
            part2 = multiple_pdfs(self.y, mu_kT, N)

            dets1 = multiple_determinants(C1)
            part2 = c2 * (1 / np.sqrt(dets1)) * part2

            vals[k] = np.sum(self.weights * (part1 - part2))
            
        # print(np.argmin(vals))
        new_input = L[np.argmin(vals), :].reshape(1, self.p)

        # import matplotlib.pyplot as plt
        # plt.scatter(L[:, 1], L[:, 2], c=vals)
        # plt.scatter(L[np.argmin(vals), :][1], L[np.argmin(vals), :][2], marker="*")
        # plt.show()

        return new_input


class lookahead(acquisition_function):
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.method = args.get("method", "ivar")
        self.nL = args.get("nL", 100)
        self.t_grid = args.get("t_grid", None)
        

    def determine_h(self):
        horizon = self.args.get("horizon", None)
        # print(horizon)
        if horizon.get("method") == "target":
            if horizon.get("previous_ratio") is None:
                horizon["previous_ratio"] = len(self.model.Z0) / len(self.model.Z)
                self.h = horizon.get("h0")
            else:
                target = horizon.get("target_ratio")
                previous_ratio = horizon.get("previous_ratio")
                current_horizon = horizon.get("h0")

                ratio = len(self.model.Z0) / len(self.model.Z)

                # Ratio increased while too small
                if ratio < target and ratio < previous_ratio:
                    self.h = max(-1, current_horizon - 1)

                # Ratio decreased while too high
                elif ratio > target and ratio > previous_ratio:
                    self.h = current_horizon + 1
                else:
                    self.h = current_horizon

                horizon["previous_ratio"] = len(self.model.Z0) / len(self.model.Z)
                horizon["h0"] = self.h

        elif horizon.get("method") == "adaptive":
            budget = np.sum(self.model.mult)

            if self.method == "imse":
                mult_star = allocate_mult(model=self.model, N=budget).astype(int)



            else:
                alloc_obj = allocate(budget, 
                                     self.model, 
                                     self.cls_func, 
                                     {"theta":self.t_ref, "weight":self.weights})

                alloc_obj.allocatereps()
                mult_star = alloc_obj.reps
            
            tab_input = mult_star - self.model.mult
            tab_input[tab_input < 0] = 0
            
            # import matplotlib.pyplot as plt
            # for label, x_count, y_count in zip(tab_input, self.model.X0[:, 1], self.model.X0[:, 2]):
            #     col = "blue"
            #     plt.annotate(
            #         label,
            #         xy=(x_count, y_count),
            #         xytext=(0, 0),
            #         textcoords="offset points",
            #         fontsize=12,
            #         color=col
            #     )
            # plt.show()
            
            rand = np.random.default_rng(self.seed)
            u, counts = np.unique(tab_input, return_counts=True)
            self.h = rand.choice(u, p=counts / counts.sum())

        elif horizon.get("h0") == -1:
            self.h = -1
        else:
            self.h = horizon.get("h0")

    def crit_eval(self, model):
        if self.method == "imse":
            value = IMSPE(model)
        else:
            value = self.total_var(model, self.x, self.t_ref)
        return value

    def evaluate(self):

        if self.args.get("long"):
            self.evaluate_vl()
        else:
            self.evaluate_vq()

    def evaluate_vl(self):

        acq_obj = eval(self.method)(
            model=self.model,
            cls_func=self.cls_func,
            args=self.args,
        )

        self.t_ref, self.weights = getattr(acq_obj, "t_ref", None), getattr(
            acq_obj, "weights", None
        )
        
        # define horizon
        self.determine_h()

        if self.h == -1:
            acq_obj.evaluate(new=True, return_pseudo=True)
            self.znew, self.new = acq_obj.znew, True
        elif self.h == 0:
            acq_obj.evaluate(new=True, return_pseudo=True)
            z_A = acq_obj.znew
            acq_obj.model.update(Xnew=z_A, Znew=acq_obj.fnew, maxit=0)
            model_A = acq_obj.model.copy()
            IVAR_A1 = self.crit_eval(model_A)

            acq_obj.model = self.model.copy()
            acq_obj.evaluate(new=False, return_pseudo=True)
            z_B = acq_obj.znew
            acq_obj.model.update(Xnew=z_B, Znew=acq_obj.fnew, maxit=0)

            model_B = acq_obj.model.copy()
            IVAR_B1 = self.crit_eval(model_B)

            if IVAR_A1 < IVAR_B1:
                self.znew, self.new = z_A, True
                return
            else:
                self.znew, self.new = z_B, False
                return
        else:
            depth = self.h + 1
            paths = np.eye(depth, dtype=int).tolist()
            inputs, values = [], []
            if self.args.get("parallel", None) is None:
                for pid, path in enumerate(paths):
                    znew, vals = self.eval_branch(acq_obj, path)
                    inputs.append(znew)
                    values.append(vals)
                    # print(np.round(vals*1000000))

                vals_id = np.argmin(values)
                new_input = inputs[vals_id][0]
                a = paths[vals_id][0]
                self.znew = new_input
                self.new = True if a == 1 else False
                return self
            else:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=len(paths))(
                    delayed(self.eval_branch)(acq_obj, path)
                    for pid, path in enumerate(paths)
                )
                inputs, vals = zip(*results)
                vals_id = np.argmin(vals)
                return inputs[vals_id][0]

    def eval_branch(self, acq_obj, action):

        acq_obj.model = self.model.copy()

        # Initialize actions
        input_action = []

        for a in action:

            new = True if a == 1 else False

            acq_obj.evaluate(new=new, return_pseudo=True)
            acq_obj.model.update(Xnew=acq_obj.znew, Znew=acq_obj.fnew, maxit=0)
            input_action.append(acq_obj.znew)

        value = self.crit_eval(acq_obj.model)

        return input_action, value

    def evaluate_vq(self):
        designs = []

        acq_obj = eval(self.method)(
            model=self.model,
            cls_func=self.cls_func,
            args=self.args,
        )
        
        self.t_ref, self.weights = getattr(acq_obj, "t_ref", None), getattr(
            acq_obj, "weights", None
        )
        
        # define horizon
        self.determine_h()

        acq_obj.evaluate(new=True, return_pseudo=True)
        z_A, f_A = acq_obj.znew, acq_obj.fnew
        path_A = [{"par": z_A, "new": True}]

        if self.h == -1:
            self.znew, self.new = z_A, True
            return
        else:
            acq_obj.model.update(Xnew=z_A, Znew=f_A, maxit=0)
            model_A = acq_obj.model.copy()
            IVAR_A1 = self.crit_eval(model_A)

            if self.h > 0:

                acq_obj.model = model_A.copy()
                for i in range(0, self.h):
                    acq_obj.evaluate(new=False, return_pseudo=True)
                    path_A.append({"par": acq_obj.znew, "new": False})
                    acq_obj.model.update(Xnew=acq_obj.znew, Znew=acq_obj.fnew, maxit=0)

                IVAR_A = self.crit_eval(acq_obj.model)

                designs.append({"input": z_A, "path": path_A, "value": IVAR_A})

            if self.h == 0:
                acq_obj.model = self.model.copy()
                acq_obj.evaluate(new=False, return_pseudo=True)
                z_B = acq_obj.znew
                acq_obj.model.update(Xnew=z_B, Znew=acq_obj.fnew, maxit=0)

                model_B = acq_obj.model.copy()
                IVAR_B1 = self.crit_eval(model_B)

                if IVAR_A1 < IVAR_B1:
                    self.znew, self.new = z_A, True
                    return
                else:
                    self.znew, self.new = z_B, False
                    return

            else:
                newmodelB = self.model.copy()
                for i in range(self.h):
                    # Choose a new replicate
                    acq_obj.model = newmodelB.copy()
                    acq_obj.evaluate(new=False, return_pseudo=True)
                    acq_obj.model.update(Xnew=acq_obj.znew, Znew=acq_obj.fnew, maxit=0)

                    if i == 0:
                        z0 = 1 * acq_obj.znew
                        path_B = []

                    path_B.append({"par": acq_obj.znew, "new": False})
                    newmodelB = acq_obj.model.copy()

                    # Choose a new design
                    acq_obj.evaluate(new=True, return_pseudo=True)
                    acq_obj.model.update(Xnew=acq_obj.znew, Znew=acq_obj.fnew, maxit=0)
                    path_C = [{"par": acq_obj.znew, "new": True}]

                    for j in range(i, self.h - 1):
                        # Remaining replicates
                        acq_obj.evaluate(new=False, return_pseudo=True)
                        acq_obj.model.update(
                            Xnew=acq_obj.znew, Znew=acq_obj.fnew, maxit=0
                        )
                        path_C.append({"par": acq_obj.znew, "new": False})

                    IVAR_C = self.crit_eval(acq_obj.model)
                    designs.append(
                        {"input": z0, "path": path_B + path_C, "value": IVAR_C}
                    )

                    if IVAR_C < IVAR_A:
                        self.znew, self.new = z0, False
                        return

                self.znew, self.new = z_A, True
                return
