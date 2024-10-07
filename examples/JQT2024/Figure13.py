from PUQ.utils import parse_arguments, save_output, read_output
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from main_deterministic import runfunction
import pandas as pd
import seaborn as sns

class covid19:
    def __init__(self):
        self.data_name   = 'covid19'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        self.truelims    = [[2.4, 3.4], [0.33, 0.99], [3.9, 4.1], [3.9, 4.1]]
        self.true_theta = [(2.9 - self.truelims[0][0])/(self.truelims[0][1] - self.truelims[0][0]), 
                           (0.66 - self.truelims[1][0])/(self.truelims[1][1] - self.truelims[1][0]), 
                           (4 - self.truelims[2][0])/(self.truelims[2][1] - self.truelims[2][0]), 
                           (4 - self.truelims[3][0])/(self.truelims[3][1] - self.truelims[3][0])]
        
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 5
        self.dx          = 1
        self.x           = None
        self.real_data   = None
        self.sigma2      = 25
        self.nodata      = True        
        

    def function(self, x, theta1, theta2, theta3, theta4):
        hosp_ad = runfunction(x, [theta1, theta2, theta3, theta4], self.truelims, point=True)
        return hosp_ad
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        H_o['f'] = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2], H['thetas'][0][3], H['thetas'][0][4])
        return H_o, persis_info
    
    def realdata(self, x, seed, isbias=False):
        
        np.random.seed(seed)
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        lm = self.truelims
        hosp_ad, daily_ad_benchmark = runfunction(None, self.true_theta, lm, point=False)

        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = np.array([daily_ad_benchmark[int(np.rint(x*188))]]) + np.random.normal(loc=0.0, scale=np.sqrt(self.sigma2), size=1) 
        self.real_data  = np.array([fevals], dtype='float64')
        
def plotresult(path, out, ex_name, w, b, rep, method, n0, nf):

    design_saved = read_output(path + out + '/', ex_name, method, w, b, rep)
    theta = design_saved._info['theta']
    f = design_saved._info['f']
    return theta, f

clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
mlist = ['P', 'p', '*', 'o', 's', 'h']
linelist = ['-', '--', '-.', ':', '-.', ':'] 
labelsb = [r'$\mathcal{A}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$']
cls_data = covid19()
des_index = np.arange(0, 189, 15)[:, None]
cls_data.realdata(des_index/188, seed=1)





def FIG11(path, outs, ex, worker, batch, rep, methods, n0, nf):
    
    for method in methods:
        theta, f = plotresult(path, outs, ex, worker, batch, rep, method, n0=n0, nf=nf)
        
        theta[:, 1] = cls_data.truelims[0][0] + theta[:, 1]*(cls_data.truelims[0][1] - cls_data.truelims[0][0])
        theta[:, 2] = cls_data.truelims[1][0] + theta[:, 2]*(cls_data.truelims[1][1] - cls_data.truelims[1][0])
        theta[:, 3] = cls_data.truelims[2][0] + theta[:, 3]*(cls_data.truelims[2][1] - cls_data.truelims[2][0])
        theta[:, 4] = cls_data.truelims[3][0] + theta[:, 4]*(cls_data.truelims[3][1] - cls_data.truelims[3][0])
        
        pdtheta = pd.DataFrame(theta[:, 1:5])
        pdtheta['color'] = np.concatenate((np.repeat('red', 50), np.repeat('gray', 150)))
        
        g = sns.pairplot(pdtheta, 
                         kind='scatter',
                         diag_kind='hist',
                         corner=True,
                         hue="color",
                         palette=['blue', 'gray'],
                         markers=["*", "X"])
        ft = 20
        from matplotlib.ticker import MaxNLocator
        from matplotlib.ticker import FormatStrFormatter
        g.axes[0, 0].axvline(x=cls_data.truelims[0][0] + cls_data.true_theta[0]*(cls_data.truelims[0][1] - cls_data.truelims[0][0]), color='red', linestyle='--', lw=2)
        g.axes[1, 1].axvline(x=cls_data.truelims[1][0] + cls_data.true_theta[1]*(cls_data.truelims[1][1] - cls_data.truelims[1][0]), color='red', linestyle='--', lw=2)
        g.axes[2, 2].axvline(x=cls_data.truelims[2][0] + cls_data.true_theta[2]*(cls_data.truelims[2][1] - cls_data.truelims[2][0]), color='red', linestyle='--', lw=2)
        g.axes[3, 3].axvline(x=cls_data.truelims[3][0] + cls_data.true_theta[3]*(cls_data.truelims[3][1] - cls_data.truelims[3][0]), color='red', linestyle='--', lw=2)
        g.axes[0, 0].set_ylabel(r'$1/\sigma_I$', fontsize=ft)
        g.axes[1, 0].set_ylabel(r'$\omega_A$', fontsize=ft)
        g.axes[2, 0].set_ylabel(r'$1/\gamma_Y$', fontsize=ft)
        g.axes[3, 0].set_ylabel(r'$1/\gamma_A$', fontsize=ft)
    
        g.axes[3, 0].set_xlabel(r'$1/\sigma_I$', fontsize=ft)
        g.axes[3, 1].set_xlabel(r'$\omega_A$', fontsize=ft)
        g.axes[3, 2].set_xlabel(r'$1/\gamma_Y$', fontsize=ft)
        g.axes[3, 3].set_xlabel(r'$1/\gamma_A$', fontsize=ft)
    
        g.axes[3, 0].tick_params(axis='both', which='major', labelsize=ft-2)
        g.axes[1, 0].tick_params(axis='both', which='major', labelsize=ft-2)
        g.axes[2, 0].tick_params(axis='both', which='major', labelsize=ft-2)
       
        g.axes[3, 1].tick_params(axis='both', which='major', labelsize=ft-2)
        g.axes[3, 2].tick_params(axis='both', which='major', labelsize=ft-2)
        g.axes[3, 3].tick_params(axis='both', which='major', labelsize=ft-2)
        
        g.axes[1, 0].yaxis.set_major_locator(MaxNLocator(3))
        g.axes[2, 0].yaxis.set_major_locator(MaxNLocator(3))
        g.axes[3, 0].yaxis.set_major_locator(MaxNLocator(3))
        g.axes[3, 0].xaxis.set_major_locator(MaxNLocator(3))
        g.axes[3, 1].xaxis.set_major_locator(MaxNLocator(3))
        g.axes[3, 2].xaxis.set_major_locator(MaxNLocator(3))
        g.axes[3, 3].xaxis.set_major_locator(MaxNLocator(3))
        
        
        g.axes[3, 0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        g.axes[3, 1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        g.axes[3, 2].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        g.axes[3, 3].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        g.axes[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        g.axes[2, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        g.axes[3, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        g._legend.remove()
        g.savefig('pairplot' + method + '.png')

    # Load the images back and display them side by side
    img1 = plt.imread('pairplotceivar.png')
    img2 = plt.imread('pairplotceivarx.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img1)
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].axis('off')

    plt.savefig('Figure12.jpg', format='jpeg', bbox_inches="tight", dpi=1000)
    plt.show()


        # g._legend.remove()
        # plt.savefig("Figure13_" + method + ".png", bbox_inches="tight")
        # plt.show()

batch = 1
worker = 2
rep = 3

fonts = 18
n0, nf = 50, 200
#path = r'/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/newPUQcovid25/' 
#path = r'/Users/ozgesurer/Desktop/covid19_bebop25/covidtrial/' 
#path = '/Users/ozgesurer/Desktop/JQT_experiments/covid19_bebop25/all/'
path = '/Users/ozgesurer/Desktop/JQT_experiments/covid19_bebop25_response/'
outs = 'covid19'
ex = 'covid19'

methods = ['ceivar', 'ceivarx'] #['ceivarx', 'ceivar', 'lhs', 'rnd']
FIG11(path, outs, ex, worker, batch, rep, methods, n0, nf)

