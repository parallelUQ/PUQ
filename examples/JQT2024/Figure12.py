
from PUQ.utils import parse_arguments, save_output, read_output
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from main_deterministic import runfunction

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


def FIG10(path, outs, ex, worker, batch, rep, methods, n0, nf):
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6)) 
    for metid, method in enumerate(methods):
        clist = ['b', 'r', 'g', 'm', 'y', 'c', 'pink', 'purple']
        mlist = ['P', 'p', '*', 'o', 's', 'h']
        linelist = ['-', '--', '-.', ':', '-.', ':'] 
        labelsb = [r'$\mathcal{A}^y$', r'$\mathcal{A}^p$', r'$\mathcal{A}^{lhs}$', r'$\mathcal{A}^{rnd}$']
        cls_data = covid19()
        des_index = np.arange(0, 189, 15)[:, None]
        cls_data.realdata(des_index/188, seed=1)
    
        theta, f = plotresult(path, outs, ex, worker, batch, rep, method, n0=n0, nf=nf)
        lm = cls_data.truelims
        hosp_ad, daily_ad_benchmark = runfunction(None, cls_data.true_theta, lm, point=False)
        
        #2020-02-28
        colors = np.repeat('blue', 189)
        index = np.arange(0, 189, 15)
        
        # Generate some random date-time data
        numdays = 222
        base = datetime.datetime(2020, 2, 28, 23, 30) 
        date_list = [base + datetime.timedelta(days=x) for x in range(0, numdays) if x >= 33]
        ft = 20
        # Set the locator
        locator = mdates.MonthLocator()  # every month
        # Specify the format - %b gives us Jan, Feb...
        fmt = mdates.DateFormatter('%b')
        axes[metid].plot(date_list, daily_ad_benchmark, c='red')
        xval = [int(np.rint(xi*188)) for xi in theta[50:, 0]]
        dateval = [date_list[xv] for xv in xval]
        axes[metid].scatter(dateval, f[50:], color='grey', marker="x", s=100)
        axes[metid].xaxis.set_major_locator(locator)
        # Specify formatter
        axes[metid].xaxis.set_major_formatter(fmt)
        axes[metid].set_ylabel('COVID-19 Hospital Admissions', fontsize=ft)
        axes[metid].tick_params(labelsize=ft-2)
    plt.savefig('Figure13.jpg', format='jpeg', bbox_inches="tight", dpi=1000)
    plt.show()

batch = 1
worker = 2
rep = 3
n0, nf = 50, 200
#path = r'/Users/ozgesurer/Desktop/GithubRepos/parallelUQ/PUQ/examples/final_results/newPUQcovid25/' 
#path = r'/Users/ozgesurer/Desktop/covid19_bebop25/covidtrial/' 
#path = '/Users/ozgesurer/Desktop/JQT_experiments/covid19_bebop25/all/'
path = '/Users/ozgesurer/Desktop/JQT_experiments/covid19_bebop25_response/'
outs = 'covid19'
ex = 'covid19'
    
methods = ['ceivar', 'ceivarx'] #['ceivarx', 'ceivar', 'lhs', 'rnd']    
FIG10(path, outs, ex, worker, batch, rep, methods, n0, nf)

# method = 'ceivar' #['ceivarx', 'ceivar', 'lhs', 'rnd']    
# FIG10(path, outs, ex, worker, batch, rep, method, n0, nf)