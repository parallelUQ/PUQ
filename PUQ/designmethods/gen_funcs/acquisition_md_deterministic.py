import numpy as np
from smt.sampling_methods import LHS
from PUQ.designmethods.support import multiple_pdfs, multiple_determinants
from hetgpy.IMSE import crit_IMSPE
import emcee

def generate_neighborhood(acq):
    N = int(acq.nL)
    sampling = LHS(xlimits=acq.tlim, random_state=acq.seed)
    L = sampling(N)
    return L

def get_pred(cL, emu, x, ttest, reps):

    cP = emu.predict(x=cL)
    var_cand = cP._info["var"] + cP._info["nugs"] / reps
    testP = emu.predict(x=ttest, thetaprime=cL)
    mu, S, cov = testP._info["mean"], testP._info["S"], testP._info["covmat"]
    mut = mu.T
    St = np.transpose(S, (2, 0, 1))

    return mut, St, cov, var_cand


class acquisition_function:
    def __init__(self, model, cls_func, args):
        self.model = model #deepcopy(model)
        self.cls_func = cls_func
        #self.persis_info = persis_info
        self.args = args
        self.x = self.cls_func.x
        self.d = self.cls_func.d
        self.p = self.cls_func.p
        self.dx = self.cls_func.dx
        self.dt = self.cls_func.dt

        self.tlim = cls_func.thetalimits
        self.y = self.cls_func.real_data
        self.Sigma = self.cls_func.obsvar
        self.Sigma3d = self.Sigma.reshape(1, self.d, self.d)
        self.detSigma = np.linalg.det(self.Sigma)
        self.twopid = (2**self.d) * (np.sqrt(np.pi) ** self.d)
        self.twopiddet = (
            (2**self.d) * (np.sqrt(np.pi) ** self.d) * np.sqrt(self.detSigma)
        )

    def acquire_new(self):
        return eval("self.evaluate")(
            self.args.get("new", True), return_pseudo=False
        )

    def gen_pred(self, model, x, t, return_id=False, return_flat=False):

        # predict at mesh
        meshPr = model.predict(x=t, thetaprime=t)
        
        # ntot, ntot x ntot, ntot
        mu, Sn, sd2 = meshPr._info["mean"], meshPr._info["S"], meshPr._info["var"]
        muT = mu.T
        S = Sn.transpose(2, 0, 1)

        return muT, S, t

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


class rnd(acquisition_function):
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.explore = args.get("explore", "both")
        self.nL = args.get("nL", 100)
        self.t_grid = args.get("t_grid", None)

    def evaluate(self, new, return_pseudo):

        new_input = self.args["prior"].rnd(1, 
                                           self.args["rand_stream"])

        self.znew, self.new = new_input, new

        if return_pseudo:
            pnew = self.model.predict(x=self.znew)
            fnew = np.array([pnew["mean"]])[None, :]
            self.fnew = fnew

        return self
    

class var(acquisition_function):
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.explore = args.get("explore", True)
        self.nL = args.get("nL", 100)

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
        vals = self.crit_pvar(model=self.model, x=self.x, t=L)
        new_input = L[np.argmax(vals), :].reshape(1, self.p)
        return new_input

class imse(acquisition_function):
    # ASK THIS FUNCTION TO DAVID
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.explore = args.get("explore", "both")
        self.nL = args.get("nL", 100)
        self.t_grid = args.get("t_grid", None)

    def evaluate(self, new, return_pseudo):

        sampling = LHS(xlimits=self.tlim, random_state=int(self.seed))
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

class exp(acquisition_function):
    # ASK THIS FUNCTION TO DAVID
    def __init__(self, model, cls_func, args):
        super().__init__(model, cls_func, args)
        self.seed = args.get("seed", None)
        self.explore = args.get("explore", "both")
        self.nL = args.get("nL", 100)
        self.t_grid = args.get("t_grid", None)

    def evaluate(self, new, return_pseudo):

        sampling = LHS(xlimits=self.tlim, random_state=int(self.seed))
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

        nm, nL = self.t_ref.shape[0], self.nL
        vals = np.zeros(nL)

        mu, S, cov, cvar = get_pred(
            cL=L, emu=self.model, x=self.x, ttest=self.t_ref, reps=1
        )
  
        V1 = S + self.Sigma3d

        q, d = self.model._info['numGPs'], mu.shape[1]

        phi = np.zeros((nm, d, d, nL))

        coef = 1 / ((2**d) * (np.sqrt(np.pi) ** d))

        for j in range(0, q):
            phi[:, j, j, :] = cov[j, :, :] ** 2 / cvar[j, :]

        for k in range(0, self.nL):
            # C1: nm x d x d
            # C2: nm x d x d
            phic = phi[:, self.x, self.x.T, k]
            C1 = (V1 + phic) * 0.5
            C2 = V1 - phic

            rpdf = multiple_pdfs(self.y, mu, C1)
            dets = multiple_determinants(C2)
            part2 = rpdf / (coef * np.sqrt(dets))

            vals[k] = np.sum(self.weights * part2)


        new_input = L[np.argmax(vals), :].reshape(1, self.p)

        return new_input

