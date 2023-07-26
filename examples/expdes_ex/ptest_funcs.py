import numpy as np

class bellcurve:
    def __init__(self):
        self.data_name   = 'bellcurve'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = 0.5
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = np.array([0.5])[:, None]
        self.dx          = len(self.x)
        self.nrep        = 1
        self.real_x      = self.x
        self.sigma2      = 0.1**2
        self.obsvar      = np.diag(np.repeat(self.sigma2/self.nrep, self.dx))

    def function(self, x, theta):
        #if x < 0.3:
        #    f = 0
        #elif x > 0.7:
        #    f = 0
        #else:
        f = np.exp(-100*(x - theta)**2) 
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info
    
    def realdata(self, seed):
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x[0], 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.function(x, self.true_theta) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
        
class sinfunc:
    def __init__(self):
        self.data_name   = 'sinfunc'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = np.pi/5
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = np.array([0.5])[:, None]
        self.dx          = len(self.x)
        self.nrep        = 1
        self.real_x      = self.x
        self.sigma2      = 0.2**2
        self.obsvar      = np.diag(np.repeat(self.sigma2/self.nrep, self.dx))

    def function(self, x, theta):
        #if x < 0.3:
        #    f = 0
        #elif x > 0.7:
        #    f = 0
        #else:
        f = np.sin(10*x - 5*theta)
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info
    
    def realdata(self, seed):
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x[0], 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.function(x, self.true_theta) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')