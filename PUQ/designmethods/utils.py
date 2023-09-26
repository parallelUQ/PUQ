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

    # Predict computer model
    emupred = emu.predict(x=x_emu, theta=xtval)
    emumean = emupred.mean()

    # Predict linear bias mean  
    bias = (obs - emumean).T
    model = LinearRegression()
    model.fit(x, bias)
    mu_bias = model.predict(x)
    diff = bias - mu_bias

    ll = (diff.T).dot(diff) 
    return ll.flatten()

def find_mle_bias(emu, x, x_emu, obs, obsvar, dx, dt, theta_limits):
    
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


def obj_covmle(parameter, args):

    x = args[0]
    biasdiff = args[1]
    nx = len(x)
    d = x.shape[1]

    diff = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            for k in range(d):
                diff[i, j] += np.abs(x[i, k] - x[j, k])
            
    covmat = np.diag(np.repeat(parameter[0], nx)) + parameter[1]*np.exp(-parameter[2]*diff)
    covmatinv = np.linalg.inv(covmat)

    ll =  np.log(np.linalg.det(covmat)) + biasdiff@covmatinv@biasdiff.T

    return ll.flatten()

def find_covparam(x, biasdiff):
    bnd = ()
    theta_init = []
    limits = [0.0001, 0.1]
    for i in range(0, 3):
        bnd += ((limits[0], limits[1]),)
        theta_init.append((limits[1])/2)
 
    opval = spo.minimize(obj_covmle,
                         theta_init,
                         method='L-BFGS-B',
                         options={'gtol': 0.01},
                         bounds=bnd,
                         args=([x, biasdiff]))                

    theta_mle = opval.x
    theta_mle = theta_mle.reshape(1, 3)
    return theta_mle[0, 0], theta_mle[0, 1], theta_mle[0, 2]


def bias_predict(emu, theta_mle, x_emu, x, true_fevals, unknowncov=False):

    nx = len(x)
    
    # Bias prediction 
    xp = np.concatenate((x, np.repeat(theta_mle, nx).reshape(nx, len(theta_mle))), axis=1)
    emupred = emu.predict(x=x_emu, theta=xp)
    mu_sim = emupred.mean()
    var_sim = emupred.var()
    bias = (true_fevals - mu_sim).T


    # Fit linear regression model
    model = LinearRegression()
    emubias = model.fit(x, bias)
    biasmean = emubias.predict(x).T
    
    if unknowncov:
        sigmae_sq, sigmab_sq, lambdap = find_covparam(x, bias.T - biasmean)

    
    class biaspred:
        def __init__(self, emubias):
            self.model = emubias
         
        def predict(self, x):
            return self.model.predict(x).T
        
        if unknowncov:
            def predictcov(self, x):
                nx = len(x)
                d = x.shape[1]
                diff = np.zeros((nx, nx))
                for i in range(nx):
                    for j in range(nx):
                        for k in range(d):
                            diff[i, j] += np.abs(x[i, k] - x[j, k])
                        
                covmat = np.diag(np.repeat(sigmae_sq, nx)) + sigmab_sq*np.exp(-lambdap*diff)
                
                return covmat

            
    biasobj = biaspred(emubias)
    return biasobj