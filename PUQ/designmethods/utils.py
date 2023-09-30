import numpy as np
from PUQ.surrogate import emulator
import pyximport
import scipy.optimize as spo
from sklearn.linear_model import LinearRegression
from PUQ.surrogatemethods.PCGPexp import  postpred, postpredbias

pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)


def fit_emulator1d(x_emu, theta, fevals):
    
    emu = emulator(x_emu, 
                   theta, 
                   fevals, 
                   method='PCGPexp')

    return emu

def compute_diff(x, 
                 parameter, 
                 nx, 
                 emu, 
                 x_emu, 
                 obs, 
                 is_bias):
    # obs : 1 x n_x
    # emumean : 1 x n_x
    # bias : n_x x 1
    xp = np.concatenate((x, np.repeat(parameter, nx).reshape(nx, len(parameter))), axis=1)
    
    # Predict computer model
    emupred = emu.predict(x=x_emu, theta=xp)
    emumean = emupred.mean()
    
    # Predict linear bias mean  
    bias = (obs - emumean).T
    if is_bias:
        model = LinearRegression().fit(x, bias)
        mu_bias = model.predict(x)
        diff = bias - mu_bias
        return diff
    else:
        return bias


def obj_mle(parameter, args):
    emu, x, x_emu, obs, obsvar, is_bias = args[0], args[1], args[2], args[3], args[4], args[5]
    nx = len(x)
    diff = compute_diff(x, parameter, nx, emu, x_emu, obs, is_bias)
    ll = diff.T@diff
    return ll.flatten()

def find_mle(emu, 
             x, 
             x_emu, 
             obs, 
             obsvar, 
             dx, 
             dt, 
             theta_limits, 
             is_bias):

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
                         args=([emu, x, x_emu, obs, obsvar, is_bias]))                

    theta_mle = opval.x
    theta_mle = theta_mle.reshape(1, dt)
    return theta_mle


def collect_data(emu, emubias, x_emu, theta_mle, dt, xmesh, xtmesh, nmesh, ytest, ptest, x, obs, obsvar, synth_info):
    
    xtrue_test = np.concatenate((xmesh, np.repeat(theta_mle, nmesh).reshape(nmesh, dt)), axis=1)
    predobj = emu.predict(x=x_emu, theta=xtrue_test)
    fmeanhat, fvarhat = predobj.mean(), predobj.var()

    if emubias == None:
        pred_error = np.mean(np.abs(fmeanhat - ytest))
        pmeanhat, pvarhat = postpred(emu._info, x, xtmesh, obs, obsvar)
        post_error = np.mean(np.abs(pmeanhat - ptest))
    else:
        bmeanhat = emubias.predict(xmesh)
        pred_error = np.mean(np.abs(fmeanhat + bmeanhat - ytest))

        bmeanhat = emubias.predict(x)
        pmeanhat, pvarhat = postpredbias(emu._info, x, xtmesh, obs, obsvar, bmeanhat)
        post_error = np.mean(np.abs(pmeanhat - ptest))

    return pred_error, post_error


def find_abs_diff(nx, d, x):
    #diff = np.zeros((nx, nx))
    #for i in range(nx):
    #    for j in range(nx):
     #       for k in range(d):
     #           diff[i, j] += np.abs(x[i, k] - x[j, k])
    diff = np.sum(np.abs(x[:, None, :] - x[None, :, :]), axis=-1)
    return diff

def gen_cov(sigmae_sq, sigmab_sq, lambdap, nx, abs_dist):
    return np.diag(np.repeat(sigmae_sq, nx)) + sigmab_sq*np.exp(-lambdap*abs_dist)
    
def obj_covmle(parameter, args):

    x, biasdiff = args[0], args[1]
    nx, d = x.shape[0], x.shape[1]

    abs_dist = find_abs_diff(nx, d, x)
    # Generate cov
    covmat = gen_cov(sigmae_sq=parameter[0], 
                     sigmab_sq=parameter[1],
                     lambdap=parameter[2],
                     nx=nx,
                     abs_dist=abs_dist)
    
    # Inverse of covmat
    covmatinv = np.linalg.inv(covmat)

    # Negative likelihood
    ll =  np.log(np.linalg.det(covmat)) + biasdiff@covmatinv@biasdiff.T
    # print(ll)
    return ll.flatten()

def find_covparam(x, biasdiff):
    bnd = ()
    theta_init = []
    limits = [0.0001, 0.5]
    for i in range(0, 3):
        bnd += ((limits[0], limits[1]),)
        theta_init.append((limits[1])/2)
 
    opval = spo.minimize(obj_covmle,
                         theta_init,
                         method='L-BFGS-B',
                         options={'gtol': 0.01},
                         bounds=bnd,
                         args=([x, biasdiff]))                

    return opval.x[0], opval.x[1], opval.x[2]


def bias_predict(emu, 
                 theta_mle, 
                 x_emu, 
                 x, 
                 obs, 
                 unknowncov=False):

    nx = len(x)
    xp = np.concatenate((x, np.repeat(theta_mle, nx).reshape(nx, len(theta_mle))), axis=1)
    
    # Predict computer model
    emupred = emu.predict(x=x_emu, theta=xp)
    emumean = emupred.mean()
    
    # Predict linear bias mean  
    bias = (obs - emumean).T
    model = LinearRegression().fit(x, bias)
    mu_bias = model.predict(x)
    diff = bias - mu_bias

    if unknowncov:
        sigmae_sq, sigmab_sq, lambdap = find_covparam(x, diff.T)

    
    class biaspred:
        def __init__(self, model):
            self.model = model
         
        def predict(self, xnew):
            return self.model.predict(xnew).T
        
        if unknowncov:
            def predictcov(self, xnew):
                nx, d = xnew.shape[0], xnew.shape[1]
                abs_dist = find_abs_diff(nx, d, xnew)
                covmat = gen_cov(sigmae_sq=sigmae_sq, 
                                 sigmab_sq=sigmab_sq,
                                 lambdap=lambdap,
                                 nx=nx,
                                 abs_dist=abs_dist)
                return covmat

            
    biasobj = biaspred(model)
    return biasobj