import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from smt.sampling_methods import LHS

def plot_EIVAR(xt, cls_data, ninit, xlim1=0, xlim2=1):
    
    plt.scatter(cls_data.x, np.repeat(cls_data.true_theta, len(cls_data.x)), marker='o', color='black')
    plt.scatter(xt[0:ninit, 0], xt[0:ninit, 1], marker='*', color='blue')
    plt.scatter(xt[:, 0][ninit:], xt[:, 1][ninit:], marker='+', color='red')
    plt.axhline(y = cls_data.true_theta, color = 'green')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()
    
    plt.hist(xt[:, 1][ninit:])
    plt.axvline(x = cls_data.true_theta, color = 'r')
    #plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel(r'$\theta$')
    plt.show()
    
    plt.hist(xt[:, 0][ninit:])
    plt.xlabel(r'x')
    plt.xlim(xlim1, xlim2)
    plt.show()

def plot_des(des, xt, n0, cls_data):
    xdes = np.array([e['x'] for e in des])
    fdes = np.array([e['feval'][0] for e in des]).T
    xu_des, xcount = np.unique(xdes, return_counts=True)
    repeatth = np.repeat(cls_data.true_theta, len(xu_des))
    for label, x_count, y_count in zip(xcount, xu_des, repeatth):
        plt.annotate(label, xy=(x_count, y_count), xytext=(x_count, y_count))
    plt.scatter(xt[0:n0, 0], xt[0:n0, 1], marker='*', color='blue')
    plt.scatter(xt[:, 0][n0:], xt[:, 1][n0:], marker='+', color='red')
    plt.axhline(y =cls_data.true_theta, color = 'black')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()
    
def plot_des_pri(xt, cls_data, ninit, nmax):
    xacq = xt[ninit:nmax, 0:2]
    tacq = xt[ninit:nmax, 2]

    plt.hist(tacq)
    plt.axvline(x =cls_data.true_theta, color = 'r')
    plt.xlabel(r'$\theta$')
    plt.xlim(0, 1)
    plt.show()
    
    unq, cnt = np.unique(xacq, return_counts=True, axis=0)
    plt.scatter(unq[:, 0], unq[:, 1])
    for label, x_count, y_count in zip(cnt, unq[:, 0], unq[:, 1]):
        plt.annotate(label, xy=(x_count, y_count), xytext=(5, -5), textcoords='offset points')
    plt.show()
    
def plot_LHS(xt, cls_data):
    plt.scatter(cls_data.x, np.repeat(cls_data.true_theta, len(cls_data.x)), marker='o', color='black')
    plt.scatter(xt[:, 0], xt[:, 1], marker='*', color='blue')
    plt.axhline(y = cls_data.true_theta, color = 'green')
    plt.xlabel('x')
    plt.ylabel(r'$\theta$')
    plt.show()

def plot_post(theta, phat, ptest, phatvar):
    if theta.shape[1] == 1:
        plt.plot(theta.flatten(), phat, c='blue', linestyle='dashed')
        plt.plot(theta.flatten(), ptest, c='black')
        plt.fill_between(theta.flatten(), phat-np.sqrt(phatvar), phat+np.sqrt(phatvar), alpha=0.2)
        plt.show()
    else:
        plt.scatter(theta[:, 0], theta[:, 1], c=phat)
        plt.show()
    
def obsdata(cls_data, is_bias):
    th_vec = [0.3, 0.4, 0.5, 0.6, 0.7]
    x_vec  = (np.arange(0, 100, 1)/100)[:, None]
    fvec   = np.zeros((len(th_vec), len(x_vec)))
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for t_id, t in enumerate(th_vec):
        for x_id, x in enumerate(x_vec):
            fvec[t_id, x_id] = cls_data.function(x, t)
        plt.plot(x_vec, fvec[t_id, :], label=r'$\theta=$' + str(t), color=colors[t_id]) 

    fvec   = np.zeros(len(x_vec))
    for x_id, x in enumerate(x_vec):
        fvec[x_id] = cls_data.function(x[0], cls_data.true_theta[0]) 
    
    if is_bias:
        fvec += cls_data.bias(x_vec).flatten()
    plt.plot(x_vec, fvec)
        
    for d_id in range(cls_data.real_data.shape[1]):
        plt.scatter(cls_data.x[d_id, 0], cls_data.real_data[0, d_id], color='black')
    plt.xlabel('x')
    plt.legend()
    plt.show()

    
def create_test(cls_data, isbias=False):
    thetamesh   = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 100)[:, None]
    xmesh = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 100)[:, None]
    
    n_t = thetamesh.shape[0]
    n_x = xmesh.shape[0]
    d = len(cls_data.x)
    if cls_data.nodata:
        thetatest, ftest, ptest = None, None, None
    else:
        #xdesign_vec = np.tile(cls_data.x.flatten(), n_t)
        #thetatest   = np.concatenate((xdesign_vec[:, None], np.repeat(thetamesh, d)[:, None]), axis=1)
        xt_test = np.array([np.concatenate([xc, th]) for th in thetamesh for xc in cls_data.x])
        ftest = np.zeros((n_t, d))
        for t_id in range(n_t):
            for x_id in range(d):
                ftest[t_id, x_id] = cls_data.function(cls_data.x[x_id, 0], thetamesh[t_id, 0])
    
        ptest = np.zeros(n_t)
        for i in range(n_t):
            meanval     = ftest[i, :] 
            if isbias:
                meanval += cls_data.bias(cls_data.x).flatten()
            rnd      = sps.multivariate_normal(mean=meanval, cov=cls_data.obsvar)
            ptest[i] = rnd.pdf(cls_data.real_data)
         
    return xt_test, ftest, ptest, thetamesh, xmesh



def create_test_non(cls_data, is_bias=False):
    n_t = 100
    n_x = cls_data.x.shape[0]

    thetamesh = np.linspace(cls_data.thetalimits[2][0], cls_data.thetalimits[2][1], n_t)[:, None]

    xt_test = np.array([np.concatenate([xc, th]) for th in thetamesh for xc in cls_data.x])
    ftest = np.zeros((n_t, n_x))
    for j in range(n_t):
        for i in range(n_x):
            ftest[j, i] = cls_data.function(cls_data.x[i, 0], cls_data.x[i, 1], thetamesh[j, 0])

    if is_bias:
        biastrue = cls_data.bias(cls_data.x[:, 0], cls_data.x[:, 1])
        ptest = np.zeros(n_t)
        for j in range(n_t):
            rnd = sps.multivariate_normal(mean=ftest[j, :] + biastrue, cov=cls_data.obsvar)
            ptest[j] = rnd.pdf(cls_data.real_data)
    else:
        ptest = np.zeros(n_t)
        for j in range(n_t):
            rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
            ptest[j] = rnd.pdf(cls_data.real_data)

    
    x1 = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 20)
    x2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 20)
    X1, X2 = np.meshgrid(x1, x2)
    xmesh = np.vstack([X1.ravel(), X2.ravel()]).T
    
    return xt_test, ftest, ptest, thetamesh, xmesh


def add_result(method_name, phat, ptest, yhat, ytest, s):
    rep = {}
    rep['method'] = method_name
    rep['Posterior Error'] = np.mean(np.abs(phat - ptest))
    rep['Prediction Error'] = np.mean(np.abs(yhat - ytest))
    rep['repno'] = s
    return rep


def samplingdata(typesampling, nmax, cls_data, seed, prior_xt):

    if typesampling == 'LHS':
        sampling = LHS(xlimits=cls_data.thetalimits, random_state=seed)
        xt = sampling(nmax)
    elif typesampling == 'Random':
        xt = prior_xt.rnd(nmax, seed=seed)

    return xt

def observe_results(result, method, rep, ninit, nmax):
    

    clist = ['b', 'r', 'g', 'm']
    mlist = ['P', 'p', '*', 'o']
    linelist = ['-', '--', '-.', ':'] 
    labelsb = [r'$\mathcal{A}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$']

    fonts = 18
    for metric in ['TV', 'HD']:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5)) 
        plt.rcParams["figure.autolayout"] = True
        for mid, m in enumerate(method):
            if metric == 'TV':
                p = np.array([r['Prediction Error'][ninit:nmax] for r in result if r['method'] == m])
                meanerror = np.mean(p, axis=0)
                sderror = np.std(p, axis=0)
                axes.plot(np.arange(len(meanerror)), meanerror, label=labelsb[mid], color=clist[mid], linestyle=linelist[mid], linewidth=4)
                plt.fill_between(np.arange(len(meanerror)), meanerror-1.96*sderror/rep, meanerror+1.96*sderror/rep, color=clist[mid], alpha=0.1)
            elif metric == 'HD':
                p = np.array([r['Posterior Error'][ninit:nmax] for r in result if r['method'] == m])
                meanerror = np.mean(p, axis=0)
                sderror = np.std(p, axis=0)
                axes.plot(np.arange(len(meanerror)), meanerror, label=labelsb[mid], color=clist[mid], linestyle=linelist[mid], linewidth=4)
                plt.fill_between(np.arange(len(meanerror)), meanerror-1.96*sderror/rep, meanerror+1.96*sderror/rep, color=clist[mid], alpha=0.1)  
        axes.set_yscale('log')
        axes.set_xlabel('# of simulation evals', fontsize=fonts) 
    
        if metric == 'TV':
            axes.set_ylabel(r'${\rm MAD}^y$', fontsize=fonts) 
        elif metric == 'HD':
            axes.set_ylabel(r'${\rm MAD}^p$', fontsize=fonts) 
        axes.tick_params(axis='both', which='major', labelsize=fonts-5)
        
        axes.legend(bbox_to_anchor=(1.1, -0.2), ncol=4, fontsize=fonts, handletextpad=0.1)
        plt.show()

def create_test_highdim(cls_data, is_bias=False):
    n_t = 1500
    n_x = cls_data.x.shape[0]
    d_x = cls_data.x.shape[1]
    sampling = LHS(xlimits=cls_data.thetalimits[d_x:, :], random_state=0)
    thetamesh = sampling(n_t)

    xt_test = np.array([np.concatenate([xc, th]) for th in thetamesh for xc in cls_data.x])
    ftest = np.zeros((n_t, n_x))
    for j in range(n_t):
        for i in range(n_x):
            if d_x == 2:
                ftest[j, i] = cls_data.function(cls_data.x[i, 0], cls_data.x[i, 1], 
                                                thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2],
                                                thetamesh[j, 3], thetamesh[j, 4], thetamesh[j, 5],
                                                thetamesh[j, 6], thetamesh[j, 7], thetamesh[j, 8], thetamesh[j, 9])
            elif d_x == 6:
                ftest[j, i] = cls_data.function(cls_data.x[i, 0], cls_data.x[i, 1], cls_data.x[i, 2], 
                                                cls_data.x[i, 3], cls_data.x[i, 4], cls_data.x[i, 5], 
                                                thetamesh[j, 0], thetamesh[j, 1], thetamesh[j, 2],
                                                thetamesh[j, 3], thetamesh[j, 4], thetamesh[j, 5])
            elif d_x == 10:
                ftest[j, i] = cls_data.function(cls_data.x[i, 0], cls_data.x[i, 1], cls_data.x[i, 2], 
                                                cls_data.x[i, 3], cls_data.x[i, 4], cls_data.x[i, 5], 
                                                cls_data.x[i, 6], cls_data.x[i, 7], cls_data.x[i, 8],
                                                cls_data.x[i, 9], thetamesh[j, 0], thetamesh[j, 1])

    ptest = np.zeros(n_t)
    for j in range(n_t):
        rnd = sps.multivariate_normal(mean=ftest[j, :], cov=cls_data.obsvar)
        ptest[j] = rnd.pdf(cls_data.real_data)
    sampling = LHS(xlimits=cls_data.thetalimits[0:d_x, :], random_state=0)
    xmesh = sampling(1500)
    
    return xt_test, ftest, ptest, thetamesh, xmesh