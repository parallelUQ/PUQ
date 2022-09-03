import numpy as np
import time
from threading import Event
import scipy.stats as sps
import os

def artificial_time(persis_info, sim_specs):
    rand_stream = persis_info['rand_stream']
    simtime_parameter = sim_specs['user']['sim_time']
    run_time = rand_stream.normal(simtime_parameter, 1, 1)
    Event().wait(run_time[0])

class bfrescox:
    def __init__(self, model):

        self.data_name = 'bfrescox'
        V = 49.2849
        r = 0.9070
        a = 0.6798
        Ws = 3.3944
        rs = 1.0941
        a2 = 0.2763
        #V = 50
        #r = 0.9
        #a = 0.65
        #Ws = 3.5
        #rs = 1
        #a2 = 0.25
        parameter = [V, r, a, Ws, rs, a2]
        file = '48Ca_template.in'
        with open(file) as f:
            content = f.readlines()
        no_p = 0;
        for idx, line in enumerate(content):
            if 'XXXXX' in line:
                no_param = line.count('XXXXX')
                line_temp = line
                for i in range(no_param):
                    line_temp = line_temp.replace("XXXXX", str(parameter[no_p]), 1) 
                    no_p += 1
                content[idx] = line_temp
        f = open("frescox_temp_input.in", "a")
        f.writelines(content)
        f.close()   
        output_file = '48Ca_temp.out'
        input_file = 'frescox_temp_input.in'
        
        os.system("/Users/ozgesurer/binw/i386/frescox < frescox_temp_input.in > 48Ca_temp.out")
        #os.system("frescox < frescox_temp_input.in > 48Ca_temp.out")
        # Read outputs
        with open(output_file) as f:
            content = f.readlines()
        cross_section = [] 
        for idline, line in enumerate(content):
            if ('X-S' in line):
                cross_section.append(float(line.split()[4]))
        os.remove(input_file)
        os.remove(output_file)
        f = np.array(cross_section) 

        
        self.thetalimits = np.array([[40, 60], # V
                                     [0.7, 1.2], # r
                                     # [0.5, 0.8], # a
                                     [2.5, 4.5]]) # Ws
                                     #[0.5, 1.5],
                                     #[0.1, 0.4]])

        self.d = 181
        self.p = 3
        self.x = np.arange(0, self.d)[:, None]
        # self.real_x = np.array([[26, 31, 41, 51, 61, 71, 76, 81, 91, 101, 111, 121, 131, 141, 151]])
        # self.real_data = np.log(np.array([[1243, 887.7, 355.5, 111.5, 26.5, 10.4, 8.3, 
        #                           7.3, 17.2, 37.6, 48.7, 38.9, 32.4, 36.4, 61.9]], dtype='float64'))
        self.real_x = np.arange(0, 181, 10)[:, None] # np.arange(181)[None, :]
        self.real_data = np.log(f[self.real_x].T, dtype='float64') #\

        self.obsvar = np.diag(np.maximum(np.repeat(0.03, 181).flatten(), 0.01*np.log(f).flatten())) #np.diag(((np.log(f*0.01))).flatten()) # np.diag(np.repeat(0.03, 181)) #np.diag(((np.log(f)*0.008)).flatten()) #np.diag(np.repeat(0.1, 181)) #np.diag(((np.log(f)*0.04)**2).flatten()) ##np.diag(np.repeat(0.1, 181)) #np.diag(((np.log(f)*0.05)**2).flatten()) #np.diag(np.repeat(0.05, 181))
        self.real_data += sps.multivariate_normal.rvs(mean=np.repeat(0, 19), cov=self.obsvar[self.real_x, self.real_x.T], size=1, random_state=1)
 
        self.model = model
        if self.model == True:
            self.out = [('f', float, (self.d,))]
        else:
            self.out = [('f', float)]

    def generate_input_file(self, parameter_values):
    
        file = '48Ca_template.in'
        with open(file) as f:
            content = f.readlines()
        no_p = 0;
        for idx, line in enumerate(content):
            if 'XXXXX' in line:
                no_param = line.count('XXXXX')
                line_temp = line
                for i in range(no_param):
                    line_temp = line_temp.replace("XXXXX", str(parameter_values[no_p]), 1) 
                    no_p += 1
                content[idx] = line_temp
        f = open("frescox_temp_input.in", "a")
        f.writelines(content)
        f.close()   
        
    def function(self):

        output_file = '48Ca_temp.out'
        input_file = 'frescox_temp_input.in'
        os.system("/Users/ozgesurer/binw/i386/frescox < frescox_temp_input.in > 48Ca_temp.out")
        #os.system("frescox < frescox_temp_input.in > 48Ca_temp.out")
        # Read outputs
        with open(output_file) as f:
            content = f.readlines()
        cross_section = [] 
        for idline, line in enumerate(content):
            if ('X-S' in line):
                cross_section.append(float(line.split()[4]))
        os.remove(input_file)
        os.remove(output_file)
        for fname in os.listdir():
            if fname.startswith("fort"):
                os.remove(fname)
        f = np.log(np.array(cross_section))
        return f

    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps frescox function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        # V = H['thetas'][0][0]
        # r = H['thetas'][0][1]
        # a = H['thetas'][0][2]
        # Ws = H['thetas'][0][3]
        # rs = H['thetas'][0][4]
        # a2 = H['thetas'][0][5]
        
        V = H['thetas'][0][0]
        r = H['thetas'][0][1]
        Ws = H['thetas'][0][2]
        
        # V = 49.2849
        # r = 0.9070
        a = 0.6798
        # Ws = 3.3944
        rs = 1.0941
        a2 = 0.2763
        
        # V = 50
        # r = 0.9
        # a = 0.65
        # Ws = 3.5
        # rs = 1
        # a2 = 0.25

        parameter = [V, r, a, Ws, rs, a2]
        self.generate_input_file(parameter)
        H_o['f'] = function()
        for fname in os.listdir():
            if fname.startswith("fort"):
                os.remove(fname)
        return H_o, persis_info


class borehole:
    def __init__(self, model):
        self.data_name = 'borehole'
        self.thetalimits = np.array([[0, 1], #rw
                                    [0, 1], # r
                                    [0, 1], # Tu
                                    [0, 1], # Hu
                                    [0, 1], # Tl
                                    [0, 1], # Hl
                                    [0, 1], # L
                                    [0, 1]]) # Kw
        
        tlim             = np.array([[0.05, 0.15], #rw
                                    [100, 50000], # r
                                    [63070, 115600], # Tu
                                    [990, 1110], # Hu
                                    [63.1, 116], # Tl
                                    [700, 820], # Hl
                                    [1120, 1680], # L
                                    [9855, 12045]]) # Kw
  
        theta0 = 0.1*(self.thetalimits[:, 0] + self.thetalimits[:, 1])
        
        rw = theta0[0]*(tlim[0][1] - tlim[0][0]) + tlim[0][0]
        r = theta0[1]*(tlim[1][1] - tlim[1][0]) + tlim[1][0]
        Tu = theta0[2]*(tlim[2][1] - tlim[2][0]) + tlim[2][0]
        Hu = theta0[3]*(tlim[3][1] - tlim[3][0]) + tlim[3][0]
        Tl = theta0[4]*(tlim[4][1] - tlim[4][0]) + tlim[4][0]
        Hl = theta0[5]*(tlim[5][1] - tlim[5][0]) + tlim[5][0]
        L = theta0[6]*(tlim[6][1] - tlim[6][0]) + tlim[6][0]
        Kw = theta0[7]*(tlim[7][1] - tlim[7][0]) + tlim[7][0]
        
        frac1 = 2 * np.pi * Tu * (Hu - Hl)
        frac2a = (2*L*Tu) / (np.log(r/rw)*(rw**2)*Kw)
        
        frac2b = Tu / Tl
        frac2 = np.log(r/rw) * (1 + frac2a + frac2b)
        f = frac1 / frac2

        self.obsvar = np.array([[0.10*f]], dtype='float64')
        self.real_data = np.array([[f]], dtype='float64') 
        self.out = [('f', float)]
        self.model = model
        self.nparams = 8
        self.shape = 1

                    
    def function(self, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8):
                
        tlim             = np.array([[0.05, 0.15], #rw
                                    [100, 50000], # r
                                    [63070, 115600], # Tu
                                    [990, 1110], # Hu
                                    [63.1, 116], # Tl
                                    [700, 820], # Hl
                                    [1120, 1680], # L
                                    [9855, 12045]]) # Kw
        tlim             = np.array([[0.05, 0.15], #rw
                                    [100*0.8, 1.2*50000], # r
                                    [63070*0.8, 1.2*115600], # Tu
                                    [990*0.8, 1.2*1110], # Hu
                                    [63.1*0.8, 1.2*116], # Tl
                                    [700*0.8, 1.2*820], # Hl
                                    [1120*0.8, 1.2*1680], # L
                                    [9855*0.8, 1.2*12045]]) # Kw        
        
        rw = theta1*(tlim[0][1] - tlim[0][0]) + tlim[0][0]
        r = theta2*(tlim[1][1] - tlim[1][0]) + tlim[1][0]
        Tu = theta3*(tlim[2][1] - tlim[2][0]) + tlim[2][0]
        Hu = theta4*(tlim[3][1] - tlim[3][0]) + tlim[3][0]
        Tl = theta5*(tlim[4][1] - tlim[4][0]) + tlim[4][0]
        Hl = theta6*(tlim[5][1] - tlim[5][0]) + tlim[5][0]
        L = theta7*(tlim[6][1] - tlim[6][0]) + tlim[6][0]
        Kw = theta8*(tlim[7][1] - tlim[7][0]) + tlim[7][0]
        
        
        frac1 = 2 * np.pi * Tu * (Hu - Hl)
        frac2a = (2*L*Tu) / (np.log(r/rw)*(rw**2)*Kw)
        frac2b = Tu / Tl
        frac2 = np.log(r/rw) * (1 + frac2a + frac2b)

        f = frac1 / frac2

        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])

        rw = H['thetas'][0][0]
        r = H['thetas'][0][1]
        Tu = H['thetas'][0][2]
        Hu = H['thetas'][0][3]
        Tl = H['thetas'][0][4]
        Hl = H['thetas'][0][5]
        L = H['thetas'][0][6]
        Kw = H['thetas'][0][7]
        
        H_o['f'] = function(rw, r, Tu, Hu, Tl, Hl, L, Kw)
    
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

    def function_loglike(self, function, theta):
        obsvar = self.obsvar
        real_data = self.real_data
        m = function(theta)
        ll = np.log(sps.norm.pdf(real_data, m, np.sqrt(obsvar)))     
        return ll


        
class banana:
    def __init__(self, model):
        self.data_name   = 'banana'
        self.thetalimits = np.array([[-20, 20], [-10, 5]])
        self.obsvar      = np.array([[10**2, 0], [0, 1]]) 
        self.real_data   = np.array([[1, 3]], dtype='float64')  
        self.model       = model
        if self.model == True:
            self.out     = [('f', float, (2,))]
        else:
            self.out     = [('f', float)]
        self.p           = 2
        self.d           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2):
        f                = np.array([theta1, theta2 + 0.03*theta1**2])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        theta1          = H['thetas'][0][0]
        theta2          = H['thetas'][0][1]
        H_o['f']        = function(theta1, theta2)
        
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

class gaussian3d:
    def __init__(self, model):
        self.data_name   = 'gaussian3d'
        self.thetalimits = np.array([[-4, 4], [-4, 4], [-4, 4]])
        self.obsvar      = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]) 
        self.real_data   = np.array([[0, 0, 0]], dtype='float64')  
        self.model       = model
        if self.model == True:
            self.out     = [('f', float, (3,))]
        else:
            self.out     = [('f', float)]
        self.d           = 3
        self.p           = 3
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2, theta3):
        f                = np.array([theta1**2 + 0.5*theta1*theta2 + 0.5*theta1*theta3, 
                                     theta2**2 + 0.5*theta2*theta1 + 0.5*theta2*theta3,
                                     theta3**2 + 0.5*theta3*theta1 + 0.5*theta3*theta2])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the gaussian3d function
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        theta1          = H['thetas'][0][0]
        theta2          = H['thetas'][0][1]
        theta3          = H['thetas'][0][2]
        H_o['f']        = function(theta1, theta2, theta3)
        
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

class gaussian6d:
    def __init__(self, model):
        self.data_name   = 'gaussian6d'
        self.thetalimits = np.array([[-4, 4], [-4, 4], [-4, 4], [-4, 4], [-4, 4], [-4, 4]])
        ov               = 0.5
        self.obsvar      = np.array([[ov, 0, 0, 0, 0, 0], [0, ov, 0, 0, 0, 0], [0, 0, ov, 0, 0, 0], 
                                     [0, 0, 0, ov, 0, 0], [0, 0, 0, 0, ov, 0], [0, 0, 0, 0, 0, ov]]) 
        self.real_data   = np.array([[0, 0, 0, 0, 0, 0]], dtype='float64')  
        self.model       = model
        if self.model == True:
            self.out     = [('f', float, (6,))]
        else:
            self.out     = [('f', float)]
        self.d           = 6
        self.p           = 6
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
   
        
    def function(self, theta1, theta2, theta3, theta4, theta5, theta6):
        f = np.array([theta1**2 + 0.5*theta1*theta2 + 0.5*theta1*theta3 + 0.5*theta1*theta4 + 0.5*theta1*theta5 + 0.5*theta1*theta6, 
                      theta2**2 + 0.5*theta2*theta1 + 0.5*theta2*theta3 + 0.5*theta2*theta4 + 0.5*theta2*theta5 + 0.5*theta2*theta6,
                      theta3**2 + 0.5*theta3*theta1 + 0.5*theta3*theta2 + 0.5*theta3*theta4 + 0.5*theta3*theta5 + 0.5*theta3*theta6,
                      theta4**2 + 0.5*theta4*theta1 + 0.5*theta4*theta2 + 0.5*theta4*theta3 + 0.5*theta4*theta5 + 0.5*theta4*theta6,
                      theta5**2 + 0.5*theta5*theta1 + 0.5*theta5*theta2 + 0.5*theta5*theta3 + 0.5*theta5*theta4 + 0.5*theta5*theta6,
                      theta6**2 + 0.5*theta6*theta1 + 0.5*theta6*theta2 + 0.5*theta6*theta3 + 0.5*theta6*theta4 + 0.5*theta6*theta5])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function = sim_specs['user']['function']
        H_o      = np.zeros(1, dtype=sim_specs['out'])
        theta1   = H['thetas'][0][0]
        theta2   = H['thetas'][0][1]
        theta3   = H['thetas'][0][2]
        theta4   = H['thetas'][0][3]
        theta5   = H['thetas'][0][4]
        theta6   = H['thetas'][0][5]
        H_o['f'] = function(theta1, theta2, theta3, theta4, theta5, theta6)
        
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

class gaussian10d:
    def __init__(self, model):
        lb = -2
        ub = 2
        self.data_name   = 'gaussian10d'
        self.thetalimits = np.array([[lb, ub], [lb, ub], [lb, ub], [lb, ub], [lb, ub], 
                                     [lb, ub], [lb, ub], [lb, ub], [lb, ub], [lb, ub]])
        ov = 0.25
        self.obsvar     = np.array([[ov, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, ov, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, ov, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, ov, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, ov, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, ov, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, ov, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, ov, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, ov, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, ov]]) 
        self.real_data  = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='float64')  
        self.model      = model
        if self.model == True:
            self.out    = [('f', float, (10,))]
        else:
            self.out    = [('f', float)]
        self.d           = 10
        self.p           = 10
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10):
        f = np.array([theta1**2 + 0.5*theta1*theta2 + 0.5*theta1*theta3 + 0.5*theta1*theta4 + 0.5*theta1*theta5 + 0.5*theta1*theta6 + 0.5*theta1*theta7 + 0.5*theta1*theta8 + 0.5*theta1*theta9 + 0.5*theta1*theta10, 
                      theta2**2 + 0.5*theta2*theta1 + 0.5*theta2*theta3 + 0.5*theta2*theta4 + 0.5*theta2*theta5 + 0.5*theta2*theta6 + 0.5*theta2*theta7 + 0.5*theta2*theta8 + 0.5*theta2*theta9 + 0.5*theta2*theta10,
                      theta3**2 + 0.5*theta3*theta1 + 0.5*theta3*theta2 + 0.5*theta3*theta4 + 0.5*theta3*theta5 + 0.5*theta3*theta6 + 0.5*theta3*theta7 + 0.5*theta3*theta8 + 0.5*theta3*theta9 + 0.5*theta3*theta10,
                      theta4**2 + 0.5*theta4*theta1 + 0.5*theta4*theta2 + 0.5*theta4*theta3 + 0.5*theta4*theta5 + 0.5*theta4*theta6 + 0.5*theta4*theta7 + 0.5*theta4*theta8 + 0.5*theta4*theta9 + 0.5*theta4*theta10,
                      theta5**2 + 0.5*theta5*theta1 + 0.5*theta5*theta2 + 0.5*theta5*theta3 + 0.5*theta5*theta4 + 0.5*theta5*theta6 + 0.5*theta5*theta7 + 0.5*theta5*theta8 + 0.5*theta5*theta9 + 0.5*theta5*theta10,
                      theta6**2 + 0.5*theta6*theta1 + 0.5*theta6*theta2 + 0.5*theta6*theta3 + 0.5*theta6*theta4 + 0.5*theta6*theta5 + 0.5*theta6*theta7 + 0.5*theta6*theta8 + 0.5*theta6*theta9 + 0.5*theta6*theta10,
                      theta7**2 + 0.5*theta7*theta1 + 0.5*theta7*theta2 + 0.5*theta7*theta3 + 0.5*theta7*theta4 + 0.5*theta7*theta5 + 0.5*theta7*theta6 + 0.5*theta7*theta8 + 0.5*theta7*theta9 + 0.5*theta7*theta10,
                      theta8**2 + 0.5*theta8*theta1 + 0.5*theta8*theta2 + 0.5*theta8*theta3 + 0.5*theta8*theta4 + 0.5*theta8*theta5 + 0.5*theta8*theta6 + 0.5*theta8*theta7 + 0.5*theta8*theta9 + 0.5*theta8*theta10,
                      theta9**2 + 0.5*theta9*theta1 + 0.5*theta9*theta2 + 0.5*theta9*theta3 + 0.5*theta9*theta4 + 0.5*theta9*theta5 + 0.5*theta9*theta6 + 0.5*theta9*theta7 + 0.5*theta9*theta8 + 0.5*theta9*theta10,
                      theta10**2 + 0.5*theta10*theta1 + 0.5*theta10*theta2 + 0.5*theta10*theta3 + 0.5*theta10*theta4 + 0.5*theta10*theta5 + 0.5*theta10*theta6 + 0.5*theta10*theta7 + 0.5*theta10*theta8 + 0.5*theta10*theta9])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta1 = H['thetas'][0][0]
        theta2 = H['thetas'][0][1]
        theta3 = H['thetas'][0][2]
        theta4 = H['thetas'][0][3]
        theta5 = H['thetas'][0][4]
        theta6 = H['thetas'][0][5]
        theta7 = H['thetas'][0][6]
        theta8 = H['thetas'][0][7]
        theta9 = H['thetas'][0][8]
        theta10 = H['thetas'][0][9]
        H_o['f'] = function(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10)
        
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

    
class unidentifiable:
    def __init__(self, model):

        self.data_name   = 'unidentifiable'
        self.thetalimits = np.array([[-8, 8], [-8, 8]])
        self.obsvar      = np.array([[1/0.01, 0], [0, 1]]) 
        self.real_data   = np.array([[0, 0]], dtype='float64')  
        self.model       = model
        if self.model == True:
            self.out     = [('f', float, (2,))]
        else:
            self.out     = [('f', float)]
        self.d           = 2
        self.p           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2):
        f                = np.array([theta1, theta2])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the unidentifiable function
        """
        function         = sim_specs['user']['function']
        H_o              = np.zeros(1, dtype=sim_specs['out'])
        theta1           = H['thetas'][0][0]
        theta2           = H['thetas'][0][1]
        H_o['f']         = function(theta1, theta2)
        
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info
    
class unimodal:
    def __init__(self, model):
        self.data_name   = 'unimodal'
        self.thetalimits = np.array([[-4, 4], [-4, 4]])
        self.obsvar      = np.array([[4]], dtype='float64')
        self.real_data   = np.array([[-6]], dtype='float64')
        self.model       = model
        self.out         = [('f', float)]
        self.d           = 1
        self.p           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]
        
    def function(self, theta1, theta2):
        thetas           = np.array([theta1, theta2]).reshape((1, 2))
        S                = np.array([[1, 0.5], [0.5, 1]])
        f                = (thetas @ S) @ thetas.T
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the unimodal function
        """
        function        = sim_specs['user']['function']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        theta1          = H['thetas'][0][0]
        theta2          = H['thetas'][0][1]
        H_o['f']        = function(theta1, theta2)
        
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info
    
class bimodal:
    def __init__(self, model):

        self.data_name   = 'bimodal'
        self.thetalimits = np.array([[-6, 6], [-4, 8]])
        self.obsvar      = np.array([[1/np.sqrt(0.2), 0], [0, 1/np.sqrt(0.75)]])
        self.real_data   = np.array([[0, 2]], dtype='float64')
        self.model       = model
        if model == True:
            self.out     = [('f', float, (2,))]
        else:
            self.out     = [('f', float)]
        self.d           = 2
        self.p           = 2
        self.x           = np.arange(0, self.d)[:, None]
        self.real_x      = np.arange(0, self.d)[:, None]

    def function(self, theta1, theta2):
        f = np.array([theta2 - theta1**2, theta2 - theta1])
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the banana function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta1 = H['thetas'][0][0]
        theta2 = H['thetas'][0][1]
        H_o['f'] = function(theta1, theta2)
        
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

    
class sinlinear:
    def __init__(self, model):
        self.thetalimits = np.array([[-10, 10]])
        self.obsvar = 1**2
        self.real_data = np.array([[0]], dtype='float64') 
        self.out = [('f', float)]
        self.nparams = 1
        self.model = model
        self.shape = 1
        
    def function(self, theta):
        f = np.sin(theta) + 0.1*theta
        #f[,1] = np.sin(theta[,1])+0.1*theta[,1]
        #f[,2] = np.sin(theta[,2])+0.1*theta[,2]
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta = H['thetas'][0]
        H_o['f'] = function(theta)
    
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info


class sinlinear2:
    def __init__(self, model):
        self.thetalimits = np.array([[-20, 20]])
        self.obsvar = 1**2
        self.real_data = np.array([[0]], dtype='float64') 
        self.out = [('f', float)]
        self.nparams = 1
        self.model = model
        
    def function(self, theta):
        f = 0.2*np.sin(theta) - 0.1*theta
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta = H['thetas'][0]
        H_o['f'] = function(theta)
    
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info


class sinlinear2d:
    def __init__(self, model):
        self.thetalimits = np.array([[-10, 10], [-10, 10]])
        self.obsvar = np.array([[1, 0.95], [0.95, 1]])
        self.real_data = np.array([[0, 0]], dtype='float64') 
        if model == True:
            self.out = [('f', float, (2,))]
        else:
            self.out = [('f', float)]
        self.model = model
        self.nparams = 2
        self.shape = 2

    def function(self, theta1, theta2):
        f = np.array([0.2*np.sin(np.sqrt(theta1**2 + theta2**2)), -0.1*(np.sqrt(theta1**2 + theta2**2))])
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta1 = H['thetas'][0][0]
        theta2 = H['thetas'][0][1]
        H_o['f'] = function(theta1, theta2)
    
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info


class sinlinear2d_2:
    def __init__(self, model):
        self.thetalimits = np.array([[-20, 20], [-20, 20]])
        self.obsvar = np.array([[1]], dtype='float64')
        self.real_data = np.array([[0]], dtype='float64') 
        self.out = [('f', float)]
        self.model = model
        self.nparams = 2
        self.shape = 1
                    
    def function(self, theta1, theta2):
        thetanorm = np.sqrt(theta1**2 + theta2**2)
        f = 0.2*np.sin(thetanorm) - 0.1*(thetanorm)
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta1 = H['thetas'][0][0]
        theta2 = H['thetas'][0][1]
        H_o['f'] = function(theta1, theta2)
    
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

class gaussaian1d:
    def __init__(self, model):
        self.thetalimits = np.array([[-3, 3]])
        self.obsvar = 1**2
        self.real_data = np.array([[0]], dtype='float64') 
        self.out = [('f', float)]
        self.nparams = 1
        self.model = model
        self.shape = 1
    def function(self, theta):
        f = theta
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta = H['thetas'][0]
        H_o['f'] = function(theta)
    
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

    def posthat(self, f, sd, obsvar, real_data):
        return sps.norm.pdf(real_data, f, np.sqrt(obsvar + sd**2))

class gap1d:
    def __init__(self, model):
        self.thetalimits = np.array([[-1, 1]])
        self.obsvar = 1**2
        self.real_data = np.array([[0]], dtype='float64') 
        self.out = [('f', float)]
        self.nparams = 1
        self.model = model
        self.shape = 1
        
    def function(self, theta):
        a = 0.05
        if (theta <= a) and (theta >= -a):
            f = theta - (1/a**2)*(a**2 - theta**2)
        else:
            f = 1*theta
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta = H['thetas'][0]
        H_o['f'] = function(theta)
    
        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info

class gap2d:
    def __init__(self, model):
        self.thetalimits = np.array([[-1, 1], [-1, 1]])
        self.obsvar = np.array([[1]], dtype='float64')
        self.real_data = np.array([[0]], dtype='float64') 

        self.out = [('f', float)]
        self.nparams = 2
        self.model = model
        self.shape = 1

                
    def function(self, theta1, theta2):
        a = 0.2
        thetanorm = np.sqrt(theta1**2 + theta2**2)
        if (thetanorm <= a) and (thetanorm >= -a):
            f = thetanorm - (1/a**2)*(a**2 - thetanorm**2)
        else:
            f = 1*thetanorm
        
        return f
        
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps the sin() function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])
        theta1 = H['thetas'][0][0]
        theta2 = H['thetas'][0][1]
        H_o['f'] = function(theta1, theta2)

        # Including artificial pause
        artificial_time(persis_info, sim_specs)
        
        return H_o, persis_info