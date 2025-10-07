import numpy as np
from scipy.stats import qmc
from PUQ.designmethods.support import multiple_pdfs, multiple_determinants
from hetgpy.IMSE import crit_IMSPE, Wij, IMSPE, allocate_mult
import emcee
from copy import deepcopy


def generate_neighborhood(acq):

    rand = np.random.default_rng(acq.seed)
    neigh_type = acq.args.get("neighbor", None)

    if neigh_type == "LHS":
        N = int(acq.nL)
        sampling = qmc.LatinHypercube(d=acq.zlim.shape[0], seed=int(acq.seed))
        L = sampling.random(n=N)
    else:
        N = int(acq.nL * 0.5)
        sampling = qmc.LatinHypercube(d=acq.zlim.shape[0], seed=int(acq.seed))
        L_explore = sampling.random(n=N)

        sampling = qmc.LatinHypercube(d=acq.tlim.shape[0], seed=int(acq.seed))
        Lt = sampling.random(n=N)

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
        self.model = model  # deepcopy(model) # model.copy()
        self.cls_func = cls_func
        # self.persis_info = persis_info
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

        return eval("self.evaluate")(self.args.get("new", True), return_pseudo=False)

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
        meshPr = model.predict(x=z, thetaprime=z)

        # ntot, ntot x ntot, ntot
        # mu, Sn, sd2 = meshPr["mean"], meshPr["cov"], meshPr["sd2"]
        mu, Sn, sd2 = meshPr._info["mean"], meshPr._info["covmat"], meshPr._info["var"]

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

        L = generate_neighborhood(self)
        new_input = self.evaluate_explore(L)
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


class imse(acquisition_function):
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.explore = args.get("explore", "both")
        self.nL = args.get("nL", 100)
        self.t_grid = args.get("t_grid", None)

    def evaluate(self, new, return_pseudo):

        sampling = LHS(xlimits=self.zlim, random_state=int(self.seed))
        L = sampling(self.nL)
        new_input = self.evaluate_explore(L)
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
            sampling = qmc.LatinHypercube(d=self.tlim.shape[0], seed=int(self.seed))
            self.t_ref = sampling.random(n=500)
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

            sampling = qmc.LatinHypercube(d=self.tlim.shape[0], seed=int(self.seed))
            loc0 = sampling.random(n=nwalkers)

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

        L = generate_neighborhood(self)
        new_input = self.evaluate_explore(L)
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

        # predict at candidates
        candPr = self.model.predict(x=L, thetaprime=z)

        # nL, nL, nL x ntot
        # candvar, candnugs, candcov = candPr["sd2"], candPr["nugs"], candPr["cov"]
        candvar, candnugs, candcov = (
            candPr._info["var"],
            candPr._info["nugs"],
            candPr._info["covmat"],
        )
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
