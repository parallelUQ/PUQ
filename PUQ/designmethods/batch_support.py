import numpy as np


def rebuild_condition(complete, prev_complete, n_theta=2):  

    if (np.sum(complete) - np.sum(prev_complete) < n_theta):
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


def update_arrays(n_x, fevals, pending, complete, calc_in, obs_offset, theta_offset):
    """Unpack from calc_in into 2D (point * rows) fevals"""

    sim_id = calc_in['sim_id']
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


def load_H(H, thetas, mse, offset=0):
    """Fill inputs into H0.
    There will be num_points x num_thetas entries
    """
    n_thetas = len(thetas)
    start = offset*n_thetas

    if thetas.shape[1] < 2:
        print(H['thetas'].shape)
        print(thetas.flatten().shape)
        H['thetas'][start:start+n_thetas] = thetas#.flatten()
    else:
        H['thetas'][start:start+n_thetas] = thetas

    H['TV'][start:start+n_thetas] = np.repeat(mse, n_thetas)

    return H