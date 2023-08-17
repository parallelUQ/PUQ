import numpy as np

class bellcurve:
    def __init__(self):
        self.data_name   = 'bellcurve'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = np.array([0.5])  
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.dx          = 1
        self.des = []
        self.nrep        = 1
        self.sigma2      = 0.1**2
        self.nodata      = True
        self.real_data  = None
        self.x          = None 
        
    def realvar(self, x):
        obsvar = np.zeros(x.shape)
        obsvar[x >= 0] = self.sigma2 
        return obsvar.ravel()
    
    def function(self, x, theta):
        f = np.exp(-100*(x - theta)**2) 
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info
    
    def realdata(self, x, seed):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                realv = self.realvar(x[0])
                fv              = self.genobsdata(x, self.true_theta)
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
    
    def genobsdata(self, x, sigma2):
        realv = self.realvar(x[0])
        return self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(realv), 1) 
        
class multicurve:
    def __init__(self):
        self.data_name   = 'multicurve'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = np.array([0.25])  
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = None
        self.real_data   = None
        self.des = []
        self.dx          = 1
        self.nrep        = 1
        self.sigma2      = 0.05**2
        self.nodata = True
    def function(self, x, theta):
        x = 12*x - 6
        if theta <= 0.5:
            f = (theta+0.5)*np.exp(-((x+3)*1)**2)
        else:
            f = theta*np.exp(-((x-3)*1)**2)
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info
    
    def realdata(self, x, seed):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                realv = self.realvar(x[0])
                fv              = self.genobsdata(x, self.true_theta) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
    
    def realvar(self, x):
        obsvar = np.zeros(x.shape)
        obsvar[x >= 0] = self.sigma2 
        obsvar = obsvar.ravel()
        return obsvar

    def genobsdata(self, x, sigma2):
        varval = self.realvar(x)
        return self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(varval), 1) 
        
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
    
    def realdata(self, x, seed):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.genobsdata(x, self.true_theta) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
    
    def realvar(self, x):
        obsvar = np.zeros(x.shape)
        obsvar[x >= 0] = self.sigma2 
        obsvar = obsvar.ravel()
        return obsvar

    def genobsdata(self, x, sigma2):
        varval = self.realvar(x)
        return self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(varval), 1) 

        

class bellcurvesimple:
    def __init__(self):
        self.data_name   = 'bellcurve'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = np.array([0.5])  
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = None
        self.real_data   = None
        self.des = []
        self.dx          = 1
        self.nrep        = 1
        self.sigma2      = 0.03**2
        self.nodata      = True  
        
    def function(self, x, theta):
        x = 6*x - 3
        f = theta*np.exp(-(x*2)**2)
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1])
        
        return H_o, persis_info
    
    def realdata(self, x, seed):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                realv = self.realvar(x[0])
                fv              = self.genobsdata(x, self.true_theta) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')

    def realvar(self, x):
        obsvar = np.zeros(x.shape)
        obsvar[x >= 0] = self.sigma2 
        obsvar = obsvar.ravel()
        return obsvar

    def genobsdata(self, x, sigma2):
        varval = self.realvar(x)
        return self.function(x[0], self.true_theta) + np.random.normal(0, np.sqrt(varval), 1) 
    
class gohbostos:
    def __init__(self):
        self.data_name   = 'gohbostos'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.true_theta  = np.array([0.5, 0.5])
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 4
        #self.x           = np.array([[0.1, 0.1], [0.5, 0.5]])
        #self.x           = np.array([[0, 0], [1, 0], [0.5, 0.5], [0, 0.5]])
        self.x           = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.x           = np.array([[0, 0], [1, 1]])
        self.dx          = len(self.x)
        self.nrep        = 1
        self.real_x      = self.x
        self.sigma2      = 0.05**2
        self.obsvar      = np.diag(np.repeat(self.sigma2/self.nrep, self.dx))

    def realvar(self, x):
        if x.ndim == 1:
            obsvar =  0.05**2
        else:
            s = len(x)
            obsvar = np.zeros(s)
            obsvar[:] = 0.05**2
        return obsvar
        
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
        
    

    
    def genobsdata(self, x, sigma2):
        return self.function(x[0], x[1], self.true_theta[0], self.true_theta[1]) + np.random.normal(0, np.sqrt(sigma2), 1) 
    
    

class nonlin:
    def __init__(self):
        self.data_name   = 'nonlin'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1]])
        self.true_theta  = np.array([0.5]) 
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 3
        x = np.linspace(0, 1, 2)
        y = np.linspace(0, 1, 2)
        self.x           = None #np.array([[xx, yy] for xx in x for yy in y])
        self.real_data   = None
        self.des = []
        self.dx          = 2
        self.nrep        = 1
        self.sigma2      = 0.05**2
        self.nodata      = True  
        
    def function(self, x1, x2, theta1):
        num = np.exp(-theta1*(x1 - 1.5*x2)**2)
        den =  np.exp(-2*theta1*(x1 + x2 - 0.7)**2)
        f = num + den
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], H['thetas'][0][1], H['thetas'][0][2])
        
        return H_o, persis_info
    
    def realdata(self, x, seed):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):

            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.genobsdata(x, None)
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
    
    def genobsdata(self, x, sigma2):
        varval = self.realvar(x)
        return self.function(x[0], x[1], self.true_theta[0]) + np.random.normal(0, np.sqrt(varval), 1) 
