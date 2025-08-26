import numpy as np
import scipy.stats as sps
from utils import heatmap, heatmap_pritam
import matplotlib.pyplot as plt
import sys

def sample_from_posterior(cls_data, seed):
    import emcee
    import scipy.stats as sps
    from smt.sampling_methods import LHS

    discard = 250
    nsteps = 1000
    thin = 30
    nwalkers = 40
    seed = int(seed)

    def log_probability(ctheta):
        if np.any((ctheta < 0) | (ctheta > 1)):
            return -np.inf
        else:
            if cls_data.dx == 1:
                feval = np.array([cls_data.function(x, *ctheta) for x in cls_data.x]).squeeze()
            elif cls_data.dx == 2:
                feval = np.array([cls_data.function(x[0], x[1], *ctheta) for x in cls_data.x]).squeeze()
            
            
            rnd = sps.multivariate_normal(mean=feval, cov=cls_data.obsvar)
            pvar = rnd.pdf(cls_data.real_data) + sys.float_info.epsilon
            return np.log(pvar)

    def sample(ndim, nwalkers, seed):
        np.random.seed(seed)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampling = LHS(
            xlimits=cls_data.zlim[cls_data.dx : cls_data.p, :], random_state=seed
        )
        loc0 = sampling(nwalkers)
        sampler.run_mcmc(initial_state=loc0, nsteps=nsteps, progress=False)
        samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        return samples

    return sample(cls_data.dt, nwalkers, seed)

def test_data_gen(cls_data, sample=False):

    # test data
    nmesh = 50
    xpl = np.linspace(cls_data.zlim[0][0], cls_data.zlim[0][1], nmesh)
    ypl = np.linspace(cls_data.zlim[1][0], cls_data.zlim[1][1], nmesh)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    tg = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
    pg = np.array(
        [
            sps.multivariate_normal(
                mean=np.array([cls_data.function(x, *t) for x in cls_data.x]),
                cov=cls_data.obsvar,
            ).pdf(cls_data.real_data)
            for t in tg
        ]
    )[:, None]

    zg = np.column_stack((np.tile(cls_data.x, (tg.shape[0], 1)), tg))
    fg = np.array([cls_data.function(*xt) for xt in zg])[:, None]
    ng = np.array([cls_data.noise(*xt) for xt in zg])[:, None]

    if sample:
        t_s = sample_from_posterior(cls_data, 1234)
        p_s = np.zeros((t_s.shape[0], 1))
        f_s = np.zeros((t_s.shape[0], 1))
        n_s = np.zeros((t_s.shape[0], 1))
        for t_id, t in enumerate(t_s):
            f_s[t_id, 0] = np.array([cls_data.function(x, t[0], t[1]) for x in cls_data.x])
            n_s[t_id, 0] = np.array([cls_data.noise(x, t[0], t[1]) for x in cls_data.x])
            rnd = sps.multivariate_normal(mean=f_s[t_id, 0], cov=cls_data.obsvar)
            p_s[t_id, 0] = rnd.pdf(cls_data.real_data)
        
        p_se = p_s + sys.float_info.epsilon
        w_s = (((1 / p_se)) / np.sum((1 / p_se))).flatten()


        heatmap(Xpl, Ypl, ng, fg, pg, t_s)

        return tg, fg, pg, zg, ng, t_s, p_s, w_s, f_s, n_s, Xpl, Ypl
    else:
        return tg, fg, pg, zg, ng, Xpl, Ypl


def test_data_gen_pri(cls_data, sample=False):

    nt = 100
    tg = np.linspace(cls_data.zlim[2][0], cls_data.zlim[2][1], nt)[:, None]
    pg = np.array(
        [
            sps.multivariate_normal(
                mean=np.array(
                    [cls_data.function(x[0], x[1], t) for x in cls_data.x]
                ).squeeze(),
                cov=cls_data.obsvar,
            ).pdf(cls_data.real_data)
            for t in tg
        ]
    )



    heatmap_pritam(cls_data)

    # (ntot, d)
    x_tiled = np.tile(cls_data.x, (tg.shape[0], 1))
    # (ntot, p-d)
    t_repeated = np.repeat(tg, cls_data.x.shape[0], axis=0)
    # (ntot, p)
    zg = np.hstack([x_tiled, t_repeated])

    fg = np.array([cls_data.function(*xt) for xt in zg])[:, None]
    ng = np.array([cls_data.noise(*xt) for xt in zg])[:, None]

    if sample:
        t_s = sample_from_posterior(cls_data, 1234)
        #t_s = np.linspace(0, 1, 900)[:, None]
        p_s = np.zeros((t_s.shape[0], 1))
        f_s = np.zeros((t_s.shape[0] * cls_data.x.shape[0], 1))
        n_s = np.zeros((t_s.shape[0] * cls_data.x.shape[0], 1))

        for t_id, t in enumerate(t_s):
            feval = np.array([cls_data.function(x[0], x[1], t) for x in cls_data.x])
            f_s[t_id*4 : (t_id + 1)*4, 0] = feval.flatten()
            n_s[t_id*4 : (t_id + 1)*4, 0] = np.array([cls_data.noise(x[0], x[1], t) for x in cls_data.x]).flatten()

            rnd = sps.multivariate_normal(mean=feval.flatten(), cov=cls_data.obsvar)
            p_s[t_id, 0] = rnd.pdf(cls_data.real_data)
        
        p_se = p_s + sys.float_info.epsilon
        w_s = (((1 / p_se)) / np.sum((1 / p_se))).flatten()

        plt.scatter(tg, pg, alpha=0.5, marker="+", color="blue", s=50)
        plt.scatter(t_s, p_s, alpha=0.1, marker="*", color="red", s=10)
        plt.show()

        return tg, fg, pg, zg, ng, t_s, p_s, w_s, f_s, n_s
    else:
        return tg, fg, pg, zg, ng
