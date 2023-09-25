import numpy as np
from PUQ.surrogate import emulator
import pyximport
import scipy.optimize as spo
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

def select_condition(complete, prev_complete, n_theta=2, n_initial=10):  
    #return False if np.sum(complete) - np.sum(prev_complete) < n_theta else True
    if (np.sum(complete) - np.sum(prev_complete) < n_theta) or (np.sum(complete) < n_initial):
        nflag = False
    else:
        nflag = True
    return nflag

def rebuild_condition(complete, prev_complete, n_theta=2, n_initial=10):  

    if (np.sum(complete) - np.sum(prev_complete) < n_theta) or (np.sum(complete) < n_initial):
        nflag = False
    else:
        nflag = True
    return nflag

def create_arrays(n_x, n_thetas):
    """Create 2D (point * rows) arrays fevals, pending and complete"""

    fevals = np.full((n_x, n_thetas), np.nan)
    pending = np.full((1, n_thetas), True)
    prev_pending = pending.copy()
    complete = np.full((1, n_thetas), False)
    prev_complete = np.full((1, n_thetas), False)


    return fevals, pending, prev_pending, complete, prev_complete

def pad_arrays(n_x, thetanew, theta, fevals, pending, prev_pending, complete, prev_complete):
    """Extend arrays to appropriate sizes."""
    n_thetanew = len(thetanew)
    theta = np.vstack((theta, thetanew))
    fevals = np.hstack((fevals, np.full((n_x, n_thetanew), np.nan)))
    pending = np.hstack((pending, np.full((1, n_thetanew), True)))
    prev_pending = np.hstack((prev_pending, np.full((1, n_thetanew), True)))
    complete = np.hstack((complete, np.full((1, n_thetanew), False)))
    prev_complete = np.hstack((prev_complete, np.full((1, n_thetanew), False)))
    
    return theta, fevals, pending, prev_pending, complete, prev_complete


def update_arrays(n_x, fevals, pending, complete, calc_in, obs_offset, theta_offset, list_id):
    """Unpack from calc_in into 2D (point * rows) fevals"""

    sim_id = calc_in['sim_id']
    list_id.append(sim_id)
    r = np.repeat(0, len(sim_id - obs_offset))
    c = sim_id - obs_offset

    if n_x < 2:
        fevals[r, c+theta_offset] = calc_in['f']
    else:
        rc = [i for j in range(len(sim_id - obs_offset)) for i in range(n_x)]
        cc = np.repeat(c, n_x)
        fevals[rc, cc] = calc_in['f'].flatten()
    
    pending[r, c+theta_offset] = False
    complete[r, c+theta_offset] = True
    return

def assign_priority(n_thetas, generated_no):
    """Assign priorities to points."""
    priority = np.arange(generated_no, generated_no + n_thetas)
    return priority

def load_H(H, thetas, mse, hd, generated_no, offset=0, set_priorities=False):
    """Fill inputs into H0.
    There will be num_points x num_thetas entries
    """
    n_thetas = len(thetas)
    start = offset*n_thetas
    
    if thetas.shape[1] < 2:
        H['thetas'][start:start+n_thetas] = thetas.flatten()
    else:
        H['thetas'][start:start+n_thetas] = thetas

    H['TV'][start:start+n_thetas] = np.repeat(mse, n_thetas)
    H['HD'][start:start+n_thetas] = np.repeat(hd, n_thetas)
    if set_priorities:
        H['priority'] = assign_priority(n_thetas, generated_no)
        
    return H


def fit_emulator(x, theta, fevals, thetalimits):
    
    idnan = np.isnan(fevals).any(axis=0).flatten()
    fevals_c = fevals[:, ~idnan]
    theta_c = theta[~idnan, :]

    emu = emulator(x, 
                   theta_c, 
                   fevals_c, 
                   method='PCGP')

    return emu

def fit_emulator1d(x_emu, theta, fevals):
    
    emu = emulator(x_emu, 
                   theta, 
                   fevals, 
                   method='PCGPexp')

    return emu

def obj_mle(parameter, args):
    emu = args[0]
    x = args[1]
    x_emu = args[2]
    obs = args[3]
    obsvar = args[4]
    nx = len(x)

    xp = np.concatenate((x, np.repeat(parameter, nx).reshape(nx, len(parameter))), axis=1)
    emupred = emu.predict(x=x_emu, theta=xp)
    mu_p = emupred.mean()
    diff = (obs.flatten() - mu_p.flatten()).reshape((nx, 1))
    ll = diff.T@diff

    return ll.flatten()

def find_mle(emu, x, x_emu, obs, obsvar, dx, dt, theta_limits):
    bnd = ()
    theta_init = []
    for i in range(dx, dx + dt):
        bnd += ((theta_limits[i][0], theta_limits[i][1]),)
        theta_init.append((theta_limits[i][0] + theta_limits[i][1])/2)
 
    opval = spo.minimize(obj_mle,
                         theta_init,
                         method='L-BFGS-B',
                         options={'gtol': 0.01},
                         bounds=bnd,
                         args=([emu, x, x_emu, obs, obsvar]))                

    theta_mle = opval.x
    theta_mle = theta_mle.reshape(1, dt)
    return theta_mle


def obj_mle_bias(parameter, args):
    
    emu = args[0]
    x = args[1]
    x_emu = args[2]
    obs = args[3]
    obsvar = args[4]
    nx = len(x)

    xtval = np.concatenate((x, np.repeat(parameter, nx).reshape(nx, len(parameter))), axis=1)

    emupred = emu.predict(x=x_emu, theta=xtval)
    emumean = emupred.mean()
    emucov = emupred.covx()
    
    bias = (obs - emumean).T

    # Fit linear regression model
    #emubias = emulator(x_emu, 
    #                   x, 
    #                   bias.T, 
    #                   method='PCGPexp')
    #diff = (bias.T - emubias.predict(x_emu, x).mean()).T

    model = LinearRegression()
    model.fit(x, bias)
    r_sq = model.score(x, bias)
    mu_bias = model.predict(x)
    diff = bias - mu_bias

    ll = (diff.T).dot(diff) 
    return ll.flatten()

def find_mle_bias(emu, x, x_emu, obs, obsvar, dx, dt, theta_limits):
    
    #parameters = np.arange(0, 1, 0.01)
    #llval = np.zeros(len(parameters))
    #for pid, p in enumerate(parameters):
    #    llval[pid] = obj_mle_bias(np.array([p]), args=([emu, x, x_emu, obs, obsvar]))
    # plt.plot(llval)
    #plt.show()
    #theta_mle = parameters[np.argmin(llval)]
    bnd = ()
    theta_init = []
    for i in range(dx, dx + dt):
        bnd += ((theta_limits[i][0], theta_limits[i][1]),)
        theta_init.append((theta_limits[i][0] + theta_limits[i][1])/2)


    opval = spo.minimize(obj_mle_bias,
                         theta_init,
                         method='L-BFGS-B',
                         options={'gtol': 0.01},
                         bounds=bnd,
                         args=([emu, x, x_emu, obs, obsvar]))                

    theta_mle = opval.x
    theta_mle = theta_mle.reshape(1, dt)
    return theta_mle