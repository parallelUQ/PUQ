
import numpy as np

class bellcurve:
    def __init__(self):
        self.data_name   = 'bellcurve'
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.true_theta  = 0.5
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[:, None] 
        self.x           = np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None] 
        self.real_x      = self.x # np.array([0.2, 0.4, 0.5, 0.6, 0.8])[:, None]
        self.theta_torun = None
        self.sigma2      = 0.2**2
        self.dx          = len(self.x)
        self.nrep        = 1
        self.obsvar      = np.diag(np.repeat(self.sigma2/self.nrep, self.dx))
        
    def function(self, x, theta):
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
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.function(x, self.true_theta) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')

class sbellcurve:
    def __init__(self):
        self.data_name   = 'sbellcurve'
        self.thetalimits = np.array([[-3, 3], [0, 1]])
        self.true_theta  = 0.5
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = np.array([-3, -1.75, 0, 1.75, 3])[:, None] 
        #self.x           = np.array([-3, -3, -1.75, -1.75, 0, 0, 1.75, 1.75, 3, 3])[:, None] 
        self.real_x      = self.x 
        self.theta_torun = None
        self.sigma2      = 0.1**2
        self.dx          = len(self.x)
        self.nrep        = 2
        self.obsvar      = np.diag(np.repeat(self.sigma2/self.nrep, self.dx))

    def function(self, x, theta):
        f = theta*np.exp(-(x*2)**2)
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
                fv              = self.function(x, self.true_theta) + np.random.normal(0, np.sqrt(self.obsvar[xid, xid]), 1) 
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
        self.x           = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[:, None] 
        self.x           = np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])[:, None] 
        self.real_x      = self.x 
        self.theta_torun = None
        self.sigma2      = 0.2**2
        self.dx          = len(self.x)
        self.nrep        = 1
        self.obsvar      = np.diag(np.repeat(self.sigma2/self.nrep, self.dx))
        
    def function(self, x, theta):
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
                fv              = self.function(x, self.true_theta) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')
    



class gohbostos:
    def __init__(self):
        self.data_name   = 'gohbostos'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.true_theta  = np.array([0.2, 0.1])
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 4
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        self.x           = np.array([[xx, yy] for xx in x for yy in y])
        #self.x = np.concatenate((np.array([[xx, yy] for xx in x for yy in y]), np.array([[xx, yy] for xx in x for yy in y])))
        self.real_x      = self.x 
        self.theta_torun = None
        self.sigma2      = 0.25**2
        self.dx          = len(self.x)
        self.nrep        = 1
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
    
class nonlin:
    def __init__(self):
        self.data_name   = 'nonlin'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1]])
        self.true_theta  = 0.5
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 3
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        self.x           = np.array([[xx, yy] for xx in x for yy in y])
        #self.x = np.concatenate((np.array([[xx, yy] for xx in x for yy in y]), np.array([[xx, yy] for xx in x for yy in y])))
        self.real_x      = self.x 
        self.theta_torun = None
        self.sigma2      = 0.1**2
        self.dx          = len(self.x)
        self.nrep        = 1
        self.obsvar      = np.diag(np.repeat(self.sigma2/self.nrep, self.dx))
        
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
    
    def realdata(self, seed):
        np.random.seed(seed)
        self.des = []
        for xid, x in enumerate(self.x):
            newd = {'x':x, 'feval':[], 'rep':self.nrep}
            for r in range(self.nrep):
                fv              = self.function(x[0], x[1], self.true_theta) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
                newd['feval'].append(fv)
            self.des.append(newd)
        
        mean_feval       = [np.mean(d['feval']) for d in self.des]
        self.real_data   = np.array([mean_feval], dtype='float64')