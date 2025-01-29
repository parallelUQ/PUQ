import numpy as np
import scipy

class unimodal:
    def __init__(self):
        self.data_name = "unimodal"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.obsvar = np.array([[0.1]], dtype="float64")
        self.real_data = None 
        self.out = [("f", float)]
        self.d = 1
        self.p = 2
        self.x = np.arange(0, self.d)[:, None] 
        self.theta_true = np.array([0.5, 0.5])

    def function(self, t1, t2):
        t1 = -10 + t1*20
        t2 = -10 + t2*20
        f = 0.26 * (t1**2 + t2**2) - 0.48 * t1 * t2
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V[0]), 1)
        f += R
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info

    def noise(self, theta):
        cov = np.array([[0.05, 0], [0, 0.05]])
        var = scipy.stats.multivariate_normal(mean=[0.85, 0.85], cov=cov)
        return 2*var.pdf(theta).reshape(self.d, theta.shape[0])

    def realdata(self, seed):

        M = np.array(self.function(self.theta_true[0], self.theta_true[1]), 
                     dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].normal(0, 
                                                  np.sqrt(self.obsvar[0]),
                                                  size=1)
            self.real_data = M + R

class bimodal:
    def __init__(self):
        self.data_name = "bimodal"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.obsvar = np.array([[0.5, 0], [0, 0.5]])
        self.theta_true = np.array([8/12, 8/12])
        self.real_data = None 
        self.out = [("f", float, (2,))]
        self.p = 2
        self.d = 2
        self.x = np.arange(0, self.d)[:, None]

    def function(self, t1, t2):
        t1 = 12*t1 -6
        t2 = 12*t2 -4
        f = np.array([np.sqrt(0.2)*(t2 - t1**2), np.sqrt(0.75)*(t2 - t1)])
        return f

    def sim_f(self, thetas, persis_info):

        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].multivariate_normal(np.array([0, 0]), 
                                                           np.diag(V.flatten()),
                                                           1)
        f += R.flatten()
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info
    
    def noise(self, theta):
        V = np.repeat(0.5 + (theta[:, 0]**2 + theta[:, 1]**2)*2, self.d)
        V = V.reshape(self.d, theta.shape[0])
        return V
    
    def realdata(self, seed):

        M  = np.array([self.function(self.theta_true[0], self.theta_true[1])], 
                      dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].multivariate_normal(mean=[0, 0], 
                                                               cov=self.obsvar, 
                                                               size=1)
            self.real_data = M + R


class banana:
    def __init__(self):
        self.data_name = "banana"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.obsvar = np.array([[0.03, 0], [0, 0.5]])
        self.theta_true = np.array([0.5, 0.75])
        self.real_data = None 
        self.out = [("f", float, (2,))]
        self.p = 2
        self.d = 2
        self.x = np.arange(0, self.d)[:, None]

    def function(self, t1, t2):
        t1 = 40*t1 - 20
        t2 = 15*t2 - 15
        f = np.array([0.03*t1, t2 + 0.06 * t1**2])
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].multivariate_normal(np.array([0, 0]), 
                                                           np.diag(V.flatten()), 
                                                           1)
        f += R.flatten() 
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info
    
    def noise(self, theta):
        
        f = self.function(theta[0, 0], theta[0, 1])
        if theta[0, 0] < 0.5:
            noise1 = 0.01*np.abs(f[0])
            noise2 = 0.01*np.abs(f[1])
        else:
            noise1 = 0.1*np.abs(f[0])
            noise2 = 0.2*np.abs(f[1])    
        return np.array([noise1, noise2]).reshape(self.d, theta.shape[0])
    
    def realdata(self, seed):

        M = np.array([self.function(self.theta_true[0], self.theta_true[1])], 
                     dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].multivariate_normal(mean=[0, 0], 
                                                               cov=self.obsvar, 
                                                               size=1)
            self.real_data = M + R
    
class sinf:
    def __init__(self):
        self.data_name = "sinf"
        self.thetalimits = np.array([[0, 1]])
        self.obsvar = np.array([[0.05]], dtype="float64")
        self.theta_true = 0.5
        self.real_data = None
        self.out = [("f", float)]
        self.d = 1
        self.p = 1
        self.x = np.arange(0, self.d)[:, None] 

    def function(self, theta1):
        return np.sin(10*theta1) 

    def sim_f(self, thetas, persis_info):
        f           = self.function(thetas[0])
        var_noise   = self.noise(np.array([thetas[0]])[None, :])
        noise       = persis_info['rand_stream'].normal(0, 
                                                        np.sqrt(var_noise[0]), 
                                                        1)
        f           += noise
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'], persis_info)
        return H_o, persis_info
        
    def realdata(self, seed):

        mean = np.array([[self.function(self.theta_true)]], dtype='float64')
        if seed is None:
            self.real_data = mean
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            noise = persis_info['rand_stream'].normal(0, 
                                                      np.sqrt(self.obsvar[0]), 
                                                      size=1)
            self.real_data = mean + noise
        
    def noise(self, theta):
        rx = (1.1 + np.sin(2*np.pi*theta))*0.05
        return rx.reshape(self.d, theta.shape[0])   
    
class branin:
    def __init__(self):
        self.data_name = "branin"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.obsvar = np.array([[10]], dtype="float64")
        self.real_data = None 
        self.out = [("f", float)]
        self.d = 1
        self.p = 2
        self.x = np.arange(0, self.d)[:, None] 
        self.theta_true = np.array([0.9613333333333334, 0.16466666666666668])

    def function(self, t1, t2):
        t1 = -5 + 15*t1
        t2 = 15*t2
        f = (t2 - (5.1/(4*np.pi**2))*(t1**2) + (5/np.pi)*t1 - 6)**2 + 10*(1 - 1/(8*np.pi))*np.cos(t1) + 10
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V[0]), 1)
        f += R
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info

    def noise(self, theta):
        cov = np.array([[0.05, 0], [0, 0.05]])
        var = scipy.stats.multivariate_normal(mean=[0.2, 0.85], cov=cov)
        return 1 + 2*var.pdf(theta).reshape(self.d, theta.shape[0])

    def realdata(self, seed):

        M = np.array(self.function(self.theta_true[0], self.theta_true[1]), 
                     dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].normal(0, 
                                                  np.sqrt(self.obsvar[0]),
                                                  size=1)
            self.real_data = M + R

class himmelblau:
    def __init__(self):

        self.data_name = "himmelblau"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-5, 5], [-5, 5]])
        self.obsvar = np.array([[100]], dtype="float64")
        self.real_data = None
        self.out = [("f", float)]
        self.p = 2
        self.d = 1
        self.x = np.arange(0, self.d)[:, None]
        self.theta_true = np.array([0.11085, 0.16])

    def function(self, theta1, theta2):

        theta1 = self.truelimits[0][0] + theta1 * (
            self.truelimits[0][1] - self.truelimits[0][0]
        )
        theta2 = self.truelimits[1][0] + theta2 * (
            self.truelimits[1][1] - self.truelimits[1][0]
        )
        f = (theta1**2 + theta2 - 11) ** 2 + (theta1 + theta2**2 - 7) ** 2
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V[0]), 1)
        f += R
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info

    def noise(self, theta):
        cov = np.array([[0.01, 0], [0, 0.01]])
        var = scipy.stats.multivariate_normal(mean=[0.5, 0.5], cov=cov)
        return 10 + 5*var.pdf(theta).reshape(self.d, theta.shape[0])


    def realdata(self, seed):

        M = np.array(self.function(self.theta_true[0], self.theta_true[1]), 
                     dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].normal(0, 
                                                  np.sqrt(self.obsvar[0]),
                                                  size=1)
            self.real_data = M + R
            
class holder:
    def __init__(self):

        self.data_name = "holder"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-10, 10], [-10, 10]])
        self.obsvar = np.array([[50]], dtype="float64")
        self.real_data = None
        self.out = [("f", float)]
        self.p = 2
        self.d = 1
        self.x = np.arange(0, self.d)[:, None]
        self.theta_true = np.array([0.1, 0.01])

    def function(self, theta1, theta2):

        theta1 = self.truelimits[0][0] + theta1 * (
            self.truelimits[0][1] - self.truelimits[0][0]
        )
        theta2 = self.truelimits[1][0] + theta2 * (
            self.truelimits[1][1] - self.truelimits[1][0]
        )
        f = -np.abs(
            np.sin(theta1)
            * np.cos(theta2)
            * np.exp(np.abs(1 - (np.sqrt(theta1**2 + theta2**2) / np.pi)))
        )
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V[0]), 1)
        f += R
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info

    def noise(self, theta):
        cov = np.array([[0.1, 0], [0, 0.1]])
        var = scipy.stats.multivariate_normal(mean=[0.5, 0.75], cov=cov)
        return 1 + 2*var.pdf(theta).reshape(self.d, theta.shape[0])
  

    def realdata(self, seed):

        M = np.array(self.function(self.theta_true[0], self.theta_true[1]), 
                     dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].normal(0, 
                                                  np.sqrt(self.obsvar[0]),
                                                  size=1)
            self.real_data = M + R
            
class easom:
    def __init__(self):

        self.data_name = "easom"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-10, 10], [-10, 10]])
        self.obsvar = np.array([[0.001]], dtype="float64")
        self.real_data = None
        self.out = [("f", float)]
        self.p = 2
        self.d = 1
        self.x = np.arange(0, self.d)[:, None]
        self.theta_true = np.array([0.75, 0.75])

    def function(self, theta1, theta2):

        theta1 = self.truelimits[0][0] + theta1 * (
            self.truelimits[0][1] - self.truelimits[0][0]
        )
        theta2 = self.truelimits[1][0] + theta2 * (
            self.truelimits[1][1] - self.truelimits[1][0]
        )
        f = (
            -np.cos(theta1)
            * np.cos(theta2)
            * np.exp(-((theta1 - np.pi) ** 2 + (theta2 - np.pi) ** 2))
        )
        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V[0]), 1)
        f += R
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info

    def noise(self, theta):
        cov = np.array([[0.1, 0], [0, 0.1]])
        var = scipy.stats.multivariate_normal(mean=[0.5, 0.75], cov=cov)
        return 1 + 2*var.pdf(theta).reshape(self.d, theta.shape[0])

    def realdata(self, seed):

        M = np.array(self.function(self.theta_true[0], self.theta_true[1]), 
                     dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].normal(0, 
                                                  np.sqrt(self.obsvar[0]),
                                                  size=1)
            self.real_data = M + R
            
            
class ackley:
    def __init__(self):

        self.data_name = "ackley"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-5, 5], [-5, 5]])
        self.obsvar = np.array([[10]], dtype="float64")
        self.real_data = None
        self.out = [("f", float)]
        self.p = 2
        self.d = 1
        self.x = np.arange(0, self.d)[:, None]
        self.theta_true = np.array([0.5, 0.5])

    def function(self, theta1, theta2):

        theta1 = self.truelimits[0][0] + theta1 * (
            self.truelimits[0][1] - self.truelimits[0][0]
        )
        theta2 = self.truelimits[1][0] + theta2 * (
            self.truelimits[1][1] - self.truelimits[1][0]
        )
        f = (
            -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (theta1**2 + theta2**2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * theta1) + np.cos(2 * np.pi * theta2)))
            + np.e
            + 20
        )

        return f

    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V[0]), 1)
        f += R
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info

    def noise(self, theta):
        cov = np.array([[0.1, 0], [0, 0.1]])
        var = scipy.stats.multivariate_normal(mean=[0.5, 0.85], cov=cov)
        return 1 + 2*var.pdf(theta).reshape(self.d, theta.shape[0])


    def realdata(self, seed):

        M = np.array(self.function(self.theta_true[0], self.theta_true[1]), 
                     dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].normal(0, 
                                                  np.sqrt(self.obsvar[0]),
                                                  size=1)
            self.real_data = M + R
            
class sphere:
    def __init__(self):

        self.data_name = "sphere"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelimits = np.array([[-5, 5], [-5, 5]])
        self.obsvar = np.array([[1]], dtype="float64")
        self.real_data = None
        self.out = [("f", float)]
        self.p = 2
        self.d = 1
        self.x = np.arange(0, self.d)[:, None]
        self.theta_true = np.array([0.5, 0.5])

    def function(self, theta1, theta2):
        theta1 = self.truelimits[0][0] + theta1 * (
            self.truelimits[0][1] - self.truelimits[0][0]
        )
        theta2 = self.truelimits[1][0] + theta2 * (
            self.truelimits[1][1] - self.truelimits[1][0]
        )
        f = theta1**2 + theta2**2
        return f


    def sim_f(self, thetas, persis_info):
        f = self.function(thetas[0], thetas[1])
        V = self.noise(np.array([thetas[0], thetas[1]])[None, :])
        R = persis_info["rand_stream"].normal(0, np.sqrt(V[0]), 1)
        f += R
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info)

        return H_o, persis_info

    def noise(self, theta):
        cov = np.array([[0.1, 0], [0, 0.1]])
        var = scipy.stats.multivariate_normal(mean=[0.5, 0.5], cov=cov)
        return 1 + 1*var.pdf(theta).reshape(self.d, theta.shape[0])


    def realdata(self, seed):

        M = np.array(self.function(self.theta_true[0], self.theta_true[1]), 
                     dtype='float64')
        if seed is None:
            self.real_data = M
        else:
            persis_info = {'rand_stream': np.random.default_rng(seed)}
            R = persis_info['rand_stream'].normal(0, 
                                                  np.sqrt(self.obsvar[0]),
                                                  size=1)
            self.real_data = M + R