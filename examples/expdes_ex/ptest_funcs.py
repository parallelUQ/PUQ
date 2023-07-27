import numpy as np

class bellcurve:
    def __init__(self):
        self.data_name   = 'bellcurve'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = np.array([0.5])  
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
        #if x < 0.4:
        #    f = 0
        #else:
        f = np.exp(-100*(x - theta)**2) 
        #elif x > 0.7:
        #    f = 0
        #else:
        
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
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                realv = self.realvar(x[0])
                fv              = self.function(x, self.true_theta) + np.random.normal(0, np.sqrt(realv), 1)  # np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
    
    def realvar(self, x):
        obsvar = np.zeros(x.shape)
        obsvar[x >= 0] = 0.1**2
        #obsvar[x >= 0.4] = 0.1**2
        #obsvar[x < 0.4] = 0.01**2
        return obsvar.ravel()

    def genobsdata(self, x, sigma2):
        return self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(sigma2), 1) 
        
    
        
class sinfunc:
    def __init__(self):
        self.data_name   = 'sinfunc'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = np.array([np.pi/5]) 
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
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
        
    
    def realvar(self, x):
        s = len(x)
        obsvar = np.zeros(s)
        obsvar[(x >= 0).ravel()] = 0.2**2
        return obsvar
    
    def genobsdata(self, x, sigma2):
        return self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(sigma2), 1) 
        
    
class gohbostos:
    def __init__(self):
        self.data_name   = 'gohbostos'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.true_theta  = np.array([0.2, 0.1])
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 4
        #self.x           = np.array([[0.1, 0.1], [0.5, 0.5]])
        self.x           = np.array([[0, 0], [1, 0], [0.5, 0.5], [0, 0.5]])
        self.dx          = len(self.x)
        self.nrep        = 1
        self.real_x      = self.x
        self.sigma2      = 0.25**2
        self.obsvar      = np.diag(np.repeat(self.sigma2/self.nrep, self.dx))


        
    def function(self, x1, x2, theta1, theta2):
        num = 1000*theta1*(x1**3) + 1900*(x1**2) + 2092*x1 + 60
        den = 100*theta2*(x1**3) + 500*(x1**2) + 4*x1 + 20
        f = (1 - np.exp(-1/(2*x2)))*(num/den)
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2], H['thetas'][0][3])
        return H_o, persis_info
    
    def realdata(self, seed):
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.function(x[0], x[1], self.true_theta[0], self.true_theta[1]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
        
    
    def realvar(self, x):
        if x.ndim == 1:
            obsvar =  0.2**2
        else:
            s = len(x)
            obsvar = np.zeros(s)
            obsvar[:] = 0.2**2
        return obsvar
    
    def genobsdata(self, x, sigma2):
        return self.function(x[0], x[1], self.true_theta[0], self.true_theta[1]) + np.random.normal(0, np.sqrt(sigma2), 1) 