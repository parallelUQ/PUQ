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
        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = self.genobsdata(x, isbias) 

        self.real_data   = np.array([fevals], dtype='float64')
    
    def genobsdata(self, x, isbias=False):
        if isbias:
            return self.function(x[0], self.true_theta[0]) + self.bias(x[0]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
        else:
            return self.function(x[0], self.true_theta[0]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 

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
        self.dx          = 2
        self.sigma2      = 0.5**2
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
        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = self.genobsdata(x, isbias) 
        
        self.real_data   = np.array([fevals], dtype='float64')  
    
    def genobsdata(self, x, isbias=False):
        if isbias:
            return self.function(x[0], x[1], self.true_theta[0]) + self.bias(x[0], x[1]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
        else:
            return self.function(x[0], x[1], self.true_theta[0]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 

    def bias(self, x1, x2):
        return -50*(np.exp(-0.2*x1 - 0.1*x2))
    
class wingweight:
    def __init__(self):
        self.data_name   = '12d'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 
                                     [0, 1], [0, 1], [0, 1], [0, 1]])
        self.true_theta  = np.array([0.5, 0.5, 0.5, 0.5]) 
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 10
        self.x           = None 
        self.real_data   = None
        self.dx          = 6
        self.sigma2      = 5
        self.nodata      = True  
        
    def function(self, x1, x2, x3, x4, x5, x6, theta1, theta2, theta3, theta4):
        # Sw, Wfw, tc, Nz, Wdg, Wp
        # A, Lam, q, lam
        A = 6 + 4*theta1
        Lam = (-10 + 20*theta2)*(np.pi/180)
        q = 16 + 29*theta3
        lam = 0.5 + 0.5*theta4

        
        Sw = 150 + 50*x1
        Wfw = 220 + 80*x2
        tc = 0.08 + 0.10*x3
        Nz = 2.5 + 3.5*x4
        Wdg = 1700 + 800*x5
        Wp = 0.025 + 0.055*x6

        
        fact1 = 0.036 * Sw**0.758 * Wfw**0.0035
        fact2 = (A / ((np.cos(Lam))**2))**0.6
        fact3 = q**0.006 * lam**0.04
        fact4 = (100*tc / np.cos(Lam))**(-0.3)
        fact5 = (Nz*Wdg)**0.49
        
        term1 = Sw * Wp
        f = fact1*fact2*fact3*fact4*fact5 + term1
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], 
                                    H['thetas'][0][1], 
                                    H['thetas'][0][2], 
                                    H['thetas'][0][3], 
                                    H['thetas'][0][4], 
                                    H['thetas'][0][5], 
                                    H['thetas'][0][6],
                                    H['thetas'][0][7],
                                    H['thetas'][0][8],
                                    H['thetas'][0][9])
        return H_o, persis_info
    
    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
        np.random.seed(seed)
        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = self.genobsdata(x, isbias) 
        
        self.real_data   = np.array([fevals], dtype='float64')  
    
    def genobsdata(self, x, isbias=False):
        return self.function(x[0], x[1], x[2], x[3], x[4], x[5], 
                              self.true_theta[0], self.true_theta[1],
                              self.true_theta[2], self.true_theta[3]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
    
# class sixd:
#     def __init__(self):
#         self.data_name   = '6d'
#         self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
#         self.true_theta  = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
#         self.out         = [('f', float)]
#         self.d           = 1
#         self.p           = 8
#         self.x           = None 
#         self.real_data   = None
#         self.dx          = 2
#         self.sigma2      = 0.01
#         self.nodata      = True  
        
#     def function(self, x1, x2, theta1, theta2, theta3, theta4, theta5, theta6):
#         f = 0.5*x1*(theta1**2 + theta2**2 + theta3**2) + 0.5*x2*(theta4**2 + theta5**2 + theta6**2)
#         return f
    
#     def sim(self, H, persis_info, sim_specs, libE_info):
#         function        = sim_specs['user']['function']
#         H_o             = np.zeros(1, dtype=sim_specs['out'])
#         H_o['f']        = function(H['thetas'][0][0], 
#                                    H['thetas'][0][1], 
#                                    H['thetas'][0][2], 
#                                    H['thetas'][0][3], 
#                                    H['thetas'][0][4], 
#                                    H['thetas'][0][5], 
#                                    H['thetas'][0][6],
#                                    H['thetas'][0][7])
#         return H_o, persis_info
    
#     def realdata(self, x, seed, isbias=False):
#         self.x = x
#         self.nodata = False
#         self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
#         np.random.seed(seed)
#         fevals = np.zeros(len(x))
#         for xid, x in enumerate(self.x):
#             fevals[xid] = self.genobsdata(x, isbias) 
        
#         self.real_data   = np.array([fevals], dtype='float64')  
    
#     def genobsdata(self, x, isbias=False):
#         return self.function(x[0], x[1], 
#                              self.true_theta[0], self.true_theta[1], self.true_theta[2], 
#                              self.true_theta[3], self.true_theta[4], self.true_theta[5]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 

# class sixx:
#     def __init__(self):
#         self.data_name   = '6d'
#         self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
#         self.true_theta  = np.array([0.5, 0.5]) 
#         self.out         = [('f', float)]
#         self.d           = 1
#         self.p           = 8
#         self.x           = None 
#         self.real_data   = None
#         self.dx          = 6
#         self.sigma2      = 0.01
#         self.nodata      = True  
        
#     def function(self, x1, x2, x3, x4, x5, x6, theta1, theta2):
#         f = 0.5*np.sqrt(theta1)*(x1 + x2**2 + np.sin(x3)) + 0.5*theta2*(x4 + x5**2 + np.sin(x6))
#         return f
    
#     def sim(self, H, persis_info, sim_specs, libE_info):
#         function        = sim_specs['user']['function']
#         H_o             = np.zeros(1, dtype=sim_specs['out'])
#         H_o['f']        = function(H['thetas'][0][0], 
#                                    H['thetas'][0][1], 
#                                    H['thetas'][0][2], 
#                                    H['thetas'][0][3], 
#                                    H['thetas'][0][4], 
#                                    H['thetas'][0][5], 
#                                    H['thetas'][0][6],
#                                    H['thetas'][0][7])
#         return H_o, persis_info
    
#     def realdata(self, x, seed, isbias=False):
#         self.x = x
#         self.nodata = False
#         self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
#         np.random.seed(seed)
#         fevals = np.zeros(len(x))
#         for xid, x in enumerate(self.x):
#             fevals[xid] = self.genobsdata(x, isbias) 
        
#         self.real_data   = np.array([fevals], dtype='float64')  
    
#     def genobsdata(self, x, isbias=False):
#         return self.function(x[0], x[1], x[2], x[3], x[4], x[5], 
#                              self.true_theta[0], self.true_theta[1]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 

class twelved:
    def __init__(self):
        self.data_name   = '12d'
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 
                                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        self.true_theta  = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 12
        self.x           = None 
        self.real_data   = None
        self.dx          = 6
        self.sigma2      = 10
        self.nodata      = True  
        
    def function(self, x1, x2, x3, x4, x5, x6, theta1, theta2, theta3, theta4, theta5, theta6):
        theta1 = -5  + 10 * theta1
        theta2 = -5  + 10 * theta2
        theta3 = -5  + 10 * theta3
        theta4 = -5  + 10 * theta4
        theta5 = -5  + 10 * theta5
        theta6 = -5  + 10 * theta6
        
        f = np.sqrt(x1 + x2 + x3 + x4 + x5 + x6)*(theta1 + theta2 + theta3 + theta4 + theta5 + theta6)**2 #theta1*(np.sin(x1) + theta2**2*x2 - (theta3 + theta4)*(x3 + x4)) - np.exp(theta5 + theta6)*(x5 + x6) #np.sqrt(theta1)*(x1 + theta2*x2**2 + theta3*np.sin(x3)) + theta4*(x4 + theta5*x5**2 + theta6*np.sin(x6))
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0][0], 
                                    H['thetas'][0][1], 
                                    H['thetas'][0][2], 
                                    H['thetas'][0][3], 
                                    H['thetas'][0][4], 
                                    H['thetas'][0][5], 
                                    H['thetas'][0][6],
                                    H['thetas'][0][7],
                                    H['thetas'][0][8],
                                    H['thetas'][0][9],
                                    H['thetas'][0][10],
                                    H['thetas'][0][11])
        return H_o, persis_info
    
    def realdata(self, x, seed, isbias=False):
        self.x = x
        self.nodata = False
        self.obsvar = np.diag(np.repeat(self.sigma2, len(self.x)))
        
        np.random.seed(seed)
        fevals = np.zeros(len(x))
        for xid, x in enumerate(self.x):
            fevals[xid] = self.genobsdata(x, isbias) 
        
        self.real_data   = np.array([fevals], dtype='float64')  
    
    def genobsdata(self, x, isbias=False):
        return self.function(x[0], x[1], x[2], x[3], x[4], x[5], 
                              self.true_theta[0], self.true_theta[1],
                              self.true_theta[2], self.true_theta[3],
                              self.true_theta[4], self.true_theta[5]) + np.random.normal(0, np.sqrt(self.sigma2), 1) 
