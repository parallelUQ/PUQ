from PUQ.utils import parse_arguments, read_output
from sir_funcs import SIR
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

yellow_colors = [
    (1, 1, 1),
    (1, 1, 0.8),  # light yellow
    (1, 1, 0.6),  
    (1, 1, 0.4),  
    (1, 1, 0.2),  
    (1, 1, 0),    # yellow    
    (1, 0.9, 0),  # dark yellow
    (1, 0.8, 0),  # yellow-orange
    (1, 0.6, 0),  # orange
    (1, 0.4, 0),  # dark orange
    (1, 0.2, 0)   # very dark orange
]
yellow_cmap = ListedColormap(yellow_colors, name='yellow')

path = "/Users/surero/Desktop/GithubRepos/PUQ/examples/4_SIR_explore/" #folderpath + ids + '_' + example + '_' + ee + '/'  + str(batch) + '/'
example = "SIR"
r = 0
batch = 8
path += str(batch) + '/'


method = "ivar"
desobj0 = read_output(path, example, method, batch+1, batch, r)
theta0 = desobj0._info['theta0']
reps0 = desobj0._info['reps0']   


method = "var"
desobj1 = read_output(path, example, method, batch+1, batch, r)
theta1 = desobj1._info['theta0']
reps1 = desobj1._info['reps0']   




nmesh = 50
nt = nmesh**2
# Create test data
nrep = 1000
persis_info = {'rand_stream': np.random.default_rng(100)}
cls_func = eval('SIR')()
cls_func.realdata(seed=r)

from smt.sampling_methods import LHS
n0 = 15
sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(r))
thetainit = sampling(n0)

xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
Xpl, Ypl = np.meshgrid(xpl, ypl)
theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T

f_test = np.zeros((nt, cls_func.d))
f_var = np.zeros((nt, cls_func.d))
for thid, th in enumerate(theta_test):
    IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
    f_test[thid, :] = np.mean(IrIdRD, axis=0)
    f_var[thid, :]  = np.var(IrIdRD, axis=0)

    

p_test = np.zeros(nmesh**2)
for thid, th in enumerate(theta_test):
    rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
    p_test[thid] = rnd.pdf(cls_func.real_data)
        
fig, ax = plt.subplots()
cs = ax.contourf(Xpl, Ypl, np.sum(f_var, axis=1).reshape(nmesh, nmesh), cmap=yellow_cmap, alpha=0.75)
cbar = fig.colorbar(cs)
cp = ax.contour(Xpl, Ypl, p_test.reshape(nmesh, nmesh), 20, cmap="coolwarm")
for label, x_count, y_count in zip(reps0, theta0[:, 0], theta0[:, 1]):
    if np.array([x_count, y_count]) in thetainit:
        plt.annotate(label, xy=(x_count, y_count), xytext=(0, 0), textcoords='offset points', fontsize=12, color='cyan')  
    else:
        plt.annotate(label, xy=(x_count, y_count), xytext=(0, 0), textcoords='offset points', fontsize=12, color='black')  
ax.set_xlabel(r"$\theta_1$", fontsize=16)
ax.set_ylabel(r"$\theta_2$", fontsize=16)
ax.tick_params(axis="both", labelsize=16)
plt.show() 

fig, ax = plt.subplots()
cs = ax.contourf(Xpl, Ypl, np.sum(f_var, axis=1).reshape(nmesh, nmesh), cmap=yellow_cmap, alpha=0.75)
cbar = fig.colorbar(cs)
cp = ax.contour(Xpl, Ypl, p_test.reshape(nmesh, nmesh), 20, cmap="coolwarm")
for label, x_count, y_count in zip(reps1, theta1[:, 0], theta1[:, 1]):
    if np.array([x_count, y_count]) in thetainit:
        plt.annotate(label, xy=(x_count, y_count), xytext=(0, 0), textcoords='offset points', fontsize=12, color='cyan')  
    else:
        plt.annotate(label, xy=(x_count, y_count), xytext=(0, 0), textcoords='offset points', fontsize=12, color='black')  
ax.set_xlabel(r"$\theta_1$", fontsize=16)
ax.set_ylabel(r"$\theta_2$", fontsize=16)
ax.tick_params(axis="both", labelsize=16)
plt.show() 

from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    multiple_pdfs,
    build_emulator,
)
x = cls_func.x
t0 = desobj0._info["theta"]
f0 = desobj0._info["f"]
emu = build_emulator(x=x, theta=t0, f=f0, pcset={"standardize": True, "latent": False})
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
plt.scatter(p_test, phat)
plt.show()

plt.hist(p_test - phat)
plt.show()

print(np.median((np.abs(p_test - phat))))

print(np.mean(np.abs((p_test - phat)/(p_test + 1))))

print( np.mean(2*np.abs(p_test - phat) / (np.abs(p_test) + np.abs(phat))))

print(sum(np.abs(np.log(1 + p_test) - np.log(1 + phat))))

print(np.mean((np.log1p(p_test) - np.log1p(phat))**2))

from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    multiple_pdfs,
    build_emulator,
)
x = cls_func.x
t0 = desobj1._info["theta"]
f0 = desobj1._info["f"]
emu = build_emulator(x=x, theta=t0, f=f0, pcset={"standardize": True, "latent": False})
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
plt.scatter(p_test, phat)
plt.show()

plt.hist(p_test - phat)
plt.show()

print(np.median(np.abs(p_test - phat)))

print(np.mean(np.abs((p_test - phat)/(p_test + 1))))


print(sum(np.abs(np.log(1 + p_test) - np.log(1 + phat))))

print(np.mean((np.log1p(p_test) - np.log1p(phat))**2))