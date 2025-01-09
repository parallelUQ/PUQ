import numpy as np
import scipy.stats as sps


class SIR:
    def __init__(self):
        self.data_name = "SIR"
        self.thetalimits = np.array([[0, 1], [0, 1]])
        self.truelims    = [[0.1, 0.3], [0.05, 0.15]]
        self.theta_true  = np.array([(0.20 - self.truelims[0][0])/(self.truelims[0][1] - self.truelims[0][0]), 
                           (0.10 - self.truelims[1][0])/(self.truelims[1][1] - self.truelims[1][0])])
        
        self.d = 3
        self.obsvar = np.zeros((self.d, self.d), dtype='float64') 
        self.obsvar[0, 0] = 0.51
        self.obsvar[1, 1] = 0.19
        self.obsvar[2, 2] = 0.58

        self.real_data = None 
        self.out = [("f", float, (self.d,))]

        self.p = 2
        self.x = np.arange(0, self.d)[:, None] 

    def simulation(self, thetas, S0=1000, I0=10, T=150, repl=1, persis_info=None):
        """
        Wraps the simulator
        """
        
        def update_sto_S(S, n_SI):
            S = S - n_SI
            return S

        def update_sto_I(I, n_SI, n_IR):
            I = I + n_SI - n_IR
            return I

        def update_sto_R(R, n_IR):
            R = R + n_IR
            return R
       
        def stochastic_sim(beta=0.2, 
                           gamma=0.1, 
                           S0=1000, 
                           I0=10, 
                           T=150,
                           repl=1, 
                           persis_info=None):
            # Initialization
            S, I, R = np.zeros((T+1, repl), dtype='int64'), np.zeros((T+1, repl), dtype='int64'), np.zeros((T+1, repl), dtype='int64')
            N       = np.zeros((T+1, repl), dtype='int64')
            S[0, :] = S0
            I[0, :] = I0
            R[0, :] = 0
            N[0, :] = S[0, :] + I[0, :] + R[0, :]
            for t in range(0, T):
                # Individual probabilities of transition
                p_SI = 1 - np.exp(-beta * I[t, :] / N[t, :])  # S to I
                p_IR = 1 - np.exp(-gamma)  # I to R

                # Draws from binomial distributions
                n_SI = persis_info["rand_stream"].binomial(S[t, :], p_SI)
                n_IR = persis_info["rand_stream"].binomial(I[t, :], p_IR)                    
                    
                S[t+1, :] = update_sto_S(S[t, :], n_SI)
                I[t+1, :] = update_sto_I(I[t, :], n_SI, n_IR)
                R[t+1, :] = update_sto_R(R[t, :], n_IR)
                N[t+1, :] = S[t+1, :] + I[t+1, :] + R[t+1, :]
                
            return S, I, R   
        
        beta_upd = self.truelims[0][0] + thetas[0]*(self.truelims[0][1] - self.truelims[0][0])
        gamma_upd = self.truelims[1][0] + thetas[1]*(self.truelims[1][1] - self.truelims[1][0])
        
        S, I, R = stochastic_sim(beta=beta_upd, 
                                 gamma=gamma_upd, 
                                 S0=S0, 
                                 I0=I0, 
                                 T=T, 
                                 repl=repl, 
                                 persis_info=persis_info)
        
        return S, I, R
        
    def sim_f(self, thetas, S0=1000, I0=10, T=150, return_all=False, repl=1, persis_info=None):

        S, I, R = self.simulation(thetas, 
                                  S0=S0,
                                  I0=I0,
                                  T=T, 
                                  repl=repl,
                                  persis_info=persis_info)

        if return_all:
            return np.concatenate([np.sqrt(np.mean(S, axis=0))[:, None], 
                                    np.sqrt(np.mean(I, axis=0))[:, None], 
                                    np.sqrt(np.mean(R, axis=0))[:, None]], axis=1)
        else:
            return np.array([np.sqrt(np.mean(S)), 
                              np.sqrt(np.mean(I)),
                              np.sqrt(np.mean(R))])


    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info=persis_info)
        return H_o, persis_info
    
    def realdata(self, seed):
        
        persis_info = {}
        persis_info['rand_stream'] = np.random.default_rng(seed)
        
        IrIdRD = self.sim_f(thetas=self.theta_true, return_all=True, repl=1000, persis_info=persis_info)

        meanf  = np.mean(IrIdRD, axis=0)
        noise  = persis_info['rand_stream'].multivariate_normal(mean=np.zeros(self.d), cov=self.obsvar, size=1).flatten() 

        if seed is None:
            self.real_data = np.array([meanf], dtype='float64')

        else:
            self.real_data = np.array([meanf + noise], dtype='float64')

            
class SEIRDS:
    def __init__(self):
        self.data_name = "SEIRDS"
        self.thetalimits = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        self.truelims    = [[0.15, 0.45], [0.15, 0.45], [0.04, 0.12], [0.06, 0.18], [0.35, 1.00], [0.005, 0.015], [0.05, 0.15]]
        self.theta_true  = np.array([(0.30 - self.truelims[0][0])/(self.truelims[0][1] - self.truelims[0][0]), 
                            (0.30 - self.truelims[1][0])/(self.truelims[1][1] - self.truelims[1][0]),
                            (0.08 - self.truelims[2][0])/(self.truelims[2][1] - self.truelims[2][0]),
                            (0.12 - self.truelims[3][0])/(self.truelims[3][1] - self.truelims[3][0]),
                            (0.70 - self.truelims[4][0])/(self.truelims[4][1] - self.truelims[4][0]),
                            (0.01 - self.truelims[5][0])/(self.truelims[5][1] - self.truelims[5][0]),
                            (0.10 - self.truelims[6][0])/(self.truelims[6][1] - self.truelims[6][0])])
        self.d = 6
        self.obsvar = np.zeros((self.d, self.d), dtype='float64') 
        self.obsvar[0, 0] = 0.48
        self.obsvar[1, 1] = 0.13
        self.obsvar[2, 2] = 0.13
        self.obsvar[3, 3] = 0.16
        self.obsvar[4, 4] = 0.27
        self.obsvar[5, 5] = 0.53
        
        self.real_data = None 
        self.out = [("f", float, (self.d,))]

        self.p = 7
        self.x = np.arange(0, self.d)[:, None] 

    def simulation(self, thetas, T=150, S0=1000, E0=10, repl=1, persis_info=None):
            
        def update_sto_S(S, n_SE, n_RS):
            S = S - n_SE + n_RS
            return S

        def update_sto_E(E, n_SE, n_EI, n_import_E):
            E = E + n_SE - n_EI + n_import_E
            return E
        
        def update_sto_Ir(Ir, n_EIr, n_IrR):
            Ir = Ir + n_EIr - n_IrR
            return Ir

        def update_sto_Id(Id, n_EId, n_IdD):
            Id = Id + n_EId - n_IdD
            return Id
        
        def update_sto_R(R, n_IrR, n_RS):
            R = R + n_IrR - n_RS
            return R

        def update_sto_D(D, n_IdD):
            D = D + n_IdD
            return D
        
        def stochastic_sim(beta=0.3, 
                           delta=0.3, 
                           gamma_R=0.08, 
                           gamma_D=0.12, 
                           mu=0.7, 
                           omega=0.01, 
                           epsilon=0.1,
                           T=150, 
                           S0=1000,
                           E0=10,
                           repl=1, 
                           persis_info=None):
            # Initialization
            S, E, Id = np.zeros((T+1, repl), dtype='int64'), np.zeros((T+1, repl), dtype='int64'), np.zeros((T+1, repl), dtype='int64')
            Ir, R, D = np.zeros((T+1, repl), dtype='int64'), np.zeros((T+1, repl), dtype='int64'), np.zeros((T+1, repl), dtype='int64')
            I        = np.zeros((T+1, repl), dtype='int64')
            N        = np.zeros((T+1, repl), dtype='int64')
            
            S[0, :] = S0
            E[0, :] = E0
            Id[0, :] = 0
            Ir[0, :] = 0
            R[0, :] = 0
            D[0, :] = 0
            I[0, :] = Id[0, :] + Ir[0, :]
            N[0, :] = S[0, :] + E[0, :] + I[0, :] + R[0, :] + D[0, :]

            p = np.array([1-mu, mu])
            
            for t in range(0, T):
                # Individual probabilities of transition
                p_SE = 1 - np.exp(-beta * I[t, :] / N[t, :])
                p_EI = 1 - np.exp(-delta)
                p_IrR = 1 - np.exp(-gamma_R) # Ir to R
                p_IdD = 1 - np.exp(-gamma_D) # Id to d
                p_RS = 1 - np.exp(-omega) # R to S

                # Draws from binomial distributions
                n_SE = persis_info["rand_stream"].binomial(S[t, :], p_SE)
                n_EI = persis_info["rand_stream"].binomial(E[t, :], p_EI)

                #n_EIrId = np.array([persis_info["rand_stream"].multinomial(n, p) for n in n_EI])
                n_EIrId = persis_info["rand_stream"].multinomial(n_EI, p) 
                
                n_EIr = n_EIrId[:, 0]
                n_EId = n_EIrId[:, 1]

                n_IrR = persis_info["rand_stream"].binomial(Ir[t, :], p_IrR)
                n_IdD = persis_info["rand_stream"].binomial(Id[t, :], p_IdD)
                n_RS = persis_info["rand_stream"].binomial(R[t, :], p_RS)
                
                n_import_E = persis_info["rand_stream"].poisson(epsilon)
                    
                
                S[t+1, :]  = update_sto_S(S[t, :], n_SE, n_RS)
                E[t+1, :]  = update_sto_E(E[t, :], n_SE, n_EI, n_import_E)
                Ir[t+1, :] = update_sto_Ir(Ir[t, :], n_EIr, n_IrR)
                Id[t+1, :] = update_sto_Id(Id[t, :], n_EId, n_IdD)
                I[t+1, :]  = Ir[t+1, :] + Id[t+1, :]
                R[t+1, :]  = update_sto_R(R[t, :], n_IrR, n_RS)
                D[t+1, :]  = update_sto_D(D[t, :], n_IdD)
                N[t+1, :]  = S[t+1, :] + E[t+1, :] + I[t+1, :] + R[t+1, :] + D[t+1, :]
            return S, E, Ir, Id, R, D   
        
        beta_u        = self.truelims[0][0] + thetas[0]*(self.truelims[0][1] - self.truelims[0][0])
        delta_u       = self.truelims[1][0] + thetas[1]*(self.truelims[1][1] - self.truelims[1][0])
        gammaR_u      = self.truelims[2][0] + thetas[2]*(self.truelims[2][1] - self.truelims[2][0])   
        gammaD_u      = self.truelims[3][0] + thetas[3]*(self.truelims[3][1] - self.truelims[3][0])   
        mu_u          = self.truelims[4][0] + thetas[4]*(self.truelims[4][1] - self.truelims[4][0])   
        omega_u       = self.truelims[5][0] + thetas[5]*(self.truelims[5][1] - self.truelims[5][0])   
        epsilon_u     = self.truelims[6][0] + thetas[6]*(self.truelims[6][1] - self.truelims[6][0])   
                   
        S, E, Ir, Id, R, D = stochastic_sim(beta=beta_u, 
                                            delta=delta_u, 
                                            gamma_R=gammaR_u, 
                                            gamma_D=gammaD_u, 
                                            mu=mu_u, 
                                            omega=omega_u, 
                                            epsilon=epsilon_u, 
                                            T=T,
                                            S0=S0,
                                            E0=E0,
                                            repl=repl,
                                            persis_info=persis_info)
        return S, E, Ir, Id, R, D         
        
    def sim_f(self, thetas, return_all=False, T=150, S0=1000, E0=10, repl=1, persis_info=None):
        """
        Wraps the simulator
        """
        S, E, Ir, Id, R, D = self.simulation(thetas, T=T, S0=S0, E0=E0, repl=repl, persis_info=persis_info)
        
        if return_all:
            return np.concatenate([np.sqrt(np.mean(S, axis=0))[:, None], 
                                    np.sqrt(np.mean(E, axis=0))[:, None], 
                                    np.sqrt(np.mean(Ir, axis=0))[:, None], 
                                    np.sqrt(np.mean(Id, axis=0))[:, None],
                                    np.sqrt(np.mean(R, axis=0))[:, None],
                                    np.sqrt(np.mean(D, axis=0))[:, None]], axis=1)
        else:
            return np.array([np.sqrt(np.mean(S)), 
                              np.sqrt(np.mean(E)),
                              np.sqrt(np.mean(Ir)), 
                              np.sqrt(np.mean(Id)),
                              np.sqrt(np.mean(R)),
                              np.sqrt(np.mean(D))])

    def sim(self, H, persis_info, sim_specs, libE_info):
        function        = sim_specs['user']['sim_f']
        H_o             = np.zeros(1, dtype=sim_specs['out'])
        H_o['f']        = function(H['thetas'][0], persis_info=persis_info)
        return H_o, persis_info
    
    def realdata(self, seed):

        persis_info = {}
        persis_info['rand_stream'] = np.random.default_rng(seed)
        
        IrIdRD = self.sim_f(thetas=self.theta_true, return_all=True, repl=1000, persis_info=persis_info)

        meanf  = np.mean(IrIdRD, axis=0)
        noise  = persis_info['rand_stream'].multivariate_normal(mean=np.zeros(self.d), cov=self.obsvar, size=1).flatten() 

        if seed is None:
            self.real_data = np.array([meanf], dtype='float64')
        else:
            self.real_data = np.array([meanf + noise], dtype='float64')
        