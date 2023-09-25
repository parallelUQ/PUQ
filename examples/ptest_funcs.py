import numpy as np

        
class sinfunc:
    def __init__(self):
        self.data_name   = 'sinfunc'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = np.array([np.pi/5]) 
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.dx          = 1
        self.x           = None
        self.real_data   = None
        self.des = []
        self.nrep        = 1
        self.sigma2      = 0.2**2
        self.nodata      = True        
        

    def function(self, x, theta):
        f = np.sin(10*x - 5*theta)
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info
    
    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv = self.genobsdata(x, self.true_theta, isbias) 
                newd['feval'].append(fv)
                newd['isreal'] = 'Yes'
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
    
    def realvar(self, x):
        obsvar = np.zeros(x.shape)
        obsvar[x >= 0] = self.sigma2 
        obsvar = obsvar.ravel()
        return obsvar

    def genobsdata(self, x, sigma2, isbias=False):
        varval = self.realvar(x)
        if isbias:
            return self.function(x[0], self.true_theta) + self.bias(x[0]) + np.random.normal(0, np.sqrt(varval), 1) 
        else:
            return self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(varval), 1) 

    def bias(self, x):
        return 1 - (1/3)*x - (2/3)*(x**2)
        
class pritam:
    def __init__(self):
        self.data_name   = 'pritam'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1]])
        self.true_theta  = np.array([0.5]) 
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 3
        self.x           = None 
        self.real_data   = None
        self.des = []
        self.dx          = 2
        self.nrep        = 1
        self.sigma2      = 0.5**2 #2
        self.nodata      = True  
        
    def function(self, x1, x2, theta1):
        f = (30 + 5*x1*np.sin(5*x1))*(6*theta1 + 1 + np.exp(-5*x2))
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2])
        
        return H_o, persis_info
    
    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):

            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.genobsdata(x, None, isbias)
                newd['feval'].append(fv)
            self.des.append(newd)
        

        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')    
    
    def realvar(self, x):
        if x.ndim == 1:
            obsvar = self.sigma2
        else:
            s = len(x)
            obsvar = np.zeros(s)
            obsvar[:] = self.sigma2
        return obsvar
    
    def genobsdata(self, x, sigma2, isbias=False):
        varval = self.realvar(x)
        if isbias:
            return self.function(x[0], x[1], self.true_theta[0]) + self.bias(x[0], x[1]) + np.random.normal(0, np.sqrt(varval), 1) 
        else:
            return self.function(x[0], x[1], self.true_theta[0]) + np.random.normal(0, np.sqrt(varval), 1) 

    def bias(self, x1, x2):
        return -50*(np.exp(-0.2*x1 - 0.1*x2))
    
    
