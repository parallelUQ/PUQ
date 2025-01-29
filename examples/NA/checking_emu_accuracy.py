from PUQ.utils import parse_arguments, read_output
from sir_funcs import SIR
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    multiple_pdfs,
    build_emulator,
)

path = "/Users/surero/Desktop/GithubRepos/PUQ/examples/4_SIR_explore/" 
example = "SIR"

batch = 8
path += str(batch) + '/'

n0 = 30
nmesh = 50

# # Create test data
nt = nmesh**2
nrep = 1000
cls_func = eval('SIR')()
xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
Xpl, Ypl = np.meshgrid(xpl, ypl)
theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T

f_test = np.zeros((nt, cls_func.d))
f_var = np.zeros((nt, cls_func.d))

persis_info = {'rand_stream': np.random.default_rng(100)}
for thid, th in enumerate(theta_test):
    IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
    f_test[thid, :] = np.mean(IrIdRD, axis=0)
    f_var[thid, :]  = np.var(IrIdRD, axis=0)
    plt.plot(np.arange(0, cls_func.d), f_test[thid, :])
IrIdRDtrue = cls_func.sim_f(thetas=cls_func.theta_true, return_all=True, repl=1000, persis_info=persis_info)
IrIdRDtrue = np.mean(IrIdRDtrue, axis=0)
plt.scatter(np.arange(0, cls_func.d), IrIdRDtrue, zorder=2)
plt.show()


xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
Xpl, Ypl = np.meshgrid(xpl, ypl)
theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T

result = []
for s in range(0, 5):
    
    cls_func = eval('SIR')()
    cls_func.realdata(seed=s)
    
    p_test = np.zeros(nmesh**2)
    for thid, th in enumerate(theta_test):
        rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
        p_test[thid] = rnd.pdf(cls_func.real_data)
                
    for method in ["ivar", "var"]:
        des = read_output(path, example, method, batch+1, batch, s)
        t = des._info['theta']
        f = des._info['f']
        x = cls_func.x
        for n in range(0, int(256/batch)):
            nid = n0 + n*batch
            te = t[0:nid, :]
            fe = f[:, 0:nid]
            
            emu = build_emulator(x=x, theta=te, f=fe, pcset={"standardize": True, "latent": False})
            testP = emu.predict(x=x, theta=theta_test)
            mu, S = testP._info["mean"], testP._info["S"]
            mut = mu.T
            St = np.transpose(S, (2, 0, 1))
            n_x = x.shape[0]
    
            # 1 x d x d
            obsvar3d = cls_func.obsvar.reshape(1, n_x, n_x)
            V1 = St + obsvar3d
            phat = np.zeros(len(theta_test))
            phat = multiple_pdfs(cls_func.real_data, mut, V1)
            
            #result.append({"metric":np.mean(2*np.abs(p_test - phat) / (np.abs(p_test) + np.abs(phat))), "method":method})
            result.append({"metric1":np.median(np.abs(p_test - phat)), "method":method, "r":s, "t":n, "batch":batch})
            result.append({"metric2":np.mean(np.abs(p_test - phat)), "method":method, "r":s, "t":n, "batch":batch})            

import pandas as pd
import seaborn as sns
df = pd.DataFrame(result)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=df, x="t", y='metric1', hue='method', style='batch', palette=['r', 'g', 'b'], ci=False, linewidth=5, ax=ax)
ax.set_yscale('log')
plt.show()
    

