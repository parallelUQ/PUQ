import numpy as np
import scipy
from numpy.linalg import cholesky, inv, det
from PUQ.surrogatemethods.covariances import cov_gen
from PUQ.designmethods.gen_funcs.acquisition_funcs_support import (
    multiple_pdfs,
    build_emulator,
    impute,
    compute_ivar
)

class allocate:
    def __init__(self, 
                 budget, 
                 emu_info, 
                 func_cls, 
                 theta_mesh, 
                 prior,
                 method='imse',
                 trace=1,
                 alloc_settings={},
                 rand_stream=None):
        
        self.budget = budget
        self.emu = emu_info
        self.x = func_cls.x
        self.theta_mesh = theta_mesh
        self.obs = func_cls.real_data
        self.obsvar = func_cls.obsvar
        self.method = method
        self.use_Ki = alloc_settings.get('use_Ki') 
        self.eps = np.sqrt(np.finfo(float).eps)
        self.trace = trace
        self.func_cls = func_cls
        self.rand_stream = rand_stream
        
        if alloc_settings.get('theta') is None:
            if alloc_settings.get('gen') is False:
                self.theta = self.emu._info['emulist'][0]['X0']
            else:
                self.theta = prior.rnd(100, None)
                self.theta = np.concatenate((self.emu._info['emulist'][0]['X0'], self.theta), axis=0)
        else:
            self.theta = alloc_settings['theta']

        if alloc_settings.get('a0') is None:
            if alloc_settings.get('gen') is False:
                self.a0 = self.emu._info['emulist'][0]['mult']
            else:
                self.a0 = np.concatenate((self.emu['emulist']._info[0]['mult'], np.repeat(0, 100)))
        else:
            self.a0 = alloc_settings['a0']            
        

    def allocatereps(self):
        
        if self.trace == 1:
            print("Allocation rule: ", self.method, " with ", self.emu._info['method'])
        
        if self.method == 'imse':
            Cinfo = self.compute_imse_weights()
            C = Cinfo.flatten()
        elif self.method == 'ivar':
            Cinfo = self.compute_ivar_weights()
            C = Cinfo.flatten()  
        elif self.method == 'var':
            Cinfo = self.compute_var_weights()
            C = Cinfo.flatten()  

        idneg = C < 0
        weight = np.sqrt(np.abs(C[idneg]))
        a_frac  = np.zeros(len(C))
        a_frac_ub = np.zeros(len(C))
        total_a   = np.sum(self.a0) + self.budget
        
        # Find an upper bound
        a_frac_ub[idneg] = total_a*weight/sum(weight)  
        a_ub = self.make_int(total_a, a_frac_ub)
        a_ub = np.maximum(0, a_ub - self.a0)

        a_frac[idneg] = self.budget*a_ub[idneg]/sum(a_ub)
        a_final = self.make_int(self.budget, a_frac)
        a_final = np.array([int(a) for a in a_final])
        if np.sum(a_final) != self.budget:
            raise ValueError("Budget constraint is not satisfied.")

        self.reps = a_final
        return 

    def compute_imse_weights(self):
        
        d = len(self.x)
        n_integ = self.theta_mesh.shape[0]
        n = self.theta.shape[0]
        
        pred_nugs = self.emu.predict(x=self.x, theta=self.theta, thetaprime=None)
        
        # d x d
        G = self.emu._info['G']
        # d x q
        B = self.emu._info['B']   
        # d x q
        GB = G @ B
        BTG = B.T @ G
        
        q = len(self.emu._info['emulist'])
        Mb = np.zeros((n, n_integ, q, q))
        for i in range(0, n):
            J = np.zeros((n, n))
            J[i, i] = 1
            for j in range(0, q):
                emuinfo = self.emu._info['emulist'][j]
                K_s = cov_gen(X1=self.theta, X2=self.theta_mesh, theta=emuinfo['theta'])
                
                if self.use_Ki:
                    Ki = emuinfo['Ki']
                else:
                    K = cov_gen(X1=self.theta, theta=emuinfo['theta'])
                    Ki = scipy.linalg.pinv(K, rcond=self.eps)
                    # Ki = scipy.linalg.pinv2(K, rcond=self.eps)
                    
                A = Ki@J@Ki
                if emuinfo['is_homGP']:
                    Mb[i, :, j, j] = -emuinfo['nu_hat']*emuinfo['g']*np.einsum('ji,jk,ki->i', K_s, A, K_s) 
                else:
                    Mb[i, :, j, j] = -(pred_nugs._info['nugs_o'][j, i])*np.einsum('ji,jk,ki->i', K_s, A, K_s)
                        
        C = np.zeros((n, 1))
        for i in range(0, n):
            for l in range(0, n_integ):
                par = GB @ Mb[i, l, :, :] @ BTG
                C[i] += np.sum(np.diag(par))       

        return C
            
            

    def compute_ivar_weights(self):
    
        d = len(self.x)
        n_integ = self.theta_mesh.shape[0]
        n = self.theta.shape[0]

        pred_nugs = self.emu.predict(x=self.x, theta=self.theta, thetaprime=None)
        pred_mesh = self.emu.predict(x=self.x, theta=self.theta_mesh, thetaprime=None)
        
        obsvar3d = self.obsvar.reshape(1, d, d) 
        # ntest x d
        mu = pred_mesh._info['mean'].T 
        S = pred_mesh._info['S'] 
        St = np.transpose(S, (2, 0, 1))
        
        # ntest x d x d
        Nb = St + 0.5*obsvar3d
        N = St + obsvar3d

        f = multiple_pdfs(self.obs, mu, Nb)
        g = multiple_pdfs(self.obs, mu, N)
        
        # ntest x 1 x d
        h = (self.obs - mu).reshape(n_integ, 1, d)
    
        # ntest x d x d
        Nbinv = inv(Nb)

        # ntest x d x d
        Ninv = inv(N)

        # ntest x 1 x d
        hNb = np.matmul(h, Nbinv)
        hN = np.matmul(h, Ninv)

        # d x d
        G = self.emu._info['G']
        # d x q
        B = self.emu._info['B']
        # d x q
        GB = G @ B
        BTG = B.T @ G

        coef = (1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det(self.obsvar))))
  
        q = len(self.emu._info['emulist'])
        Mb = np.zeros((n, n_integ, q, q))

        for i in range(0, n):
            J = np.zeros((n, n))
            J[i, i] = 1
            for j in range(0, q):
                emuinfo = self.emu._info['emulist'][j]
                K_s = cov_gen(X1=self.theta, X2=self.theta_mesh, theta=emuinfo['theta'])
                
                if self.use_Ki:
                    Ki = emuinfo['Ki']
                else:
                    K = cov_gen(X1=self.theta, theta=emuinfo['theta'])
                    Ki = scipy.linalg.pinv(K, rcond=self.eps)
  
                A = Ki@J@Ki
                if emuinfo['is_homGP']:
                    Mb[i, :, j, j] = -(emuinfo['g']*emuinfo['nu_hat'])*np.einsum('ji,jk,ki->i', K_s, A, K_s) 
                else:
                    Mb[i, :, j, j] = -(pred_nugs._info['nugs_o'][j, i])*np.einsum('ji,jk,ki->i', K_s, A, K_s) 

        dlogfdai = np.zeros((n, n_integ))
        dloggdai = np.zeros((n, n_integ))
        C = np.zeros((n, 1))
        for i in range(0, n):

            M = GB @ Mb[i, :, :, :] @ BTG
            part1f = -0.5 * np.trace(np.matmul(Nbinv, M), axis1=1, axis2=2)
            part2f = 0.5 * (np.matmul(np.matmul(hNb, M), np.transpose(hNb, (0, 2, 1))))
            part1g = -0.5 * np.trace(np.matmul(Ninv, M), axis1=1, axis2=2)
            part2g = 0.5 * (np.matmul(np.matmul(hN, M), np.transpose(hN, (0, 2, 1))))
            
            dlogfdai = part1f + part2f.flatten()
            dloggdai = part1g + part2g.flatten()
            C[i] = np.sum(coef*(f*dlogfdai) - 2*(g**2)*dloggdai)

        return C

    def compute_var_weights(self):
    
        d = len(self.x)
        pred_nugs = self.emu.predict(x=self.x, theta=self.theta, thetaprime=None)
   
        obsvar3d = self.obsvar.reshape(1, d, d)
        
        # ntest x d
        mu = pred_nugs._info['mean'].T 
        S = pred_nugs._info['S'] 
        St = np.transpose(S, (2, 0, 1))
        
        # ntest x d x d
        Nb = St + 0.5*obsvar3d
        N = St + obsvar3d

        f = multiple_pdfs(self.obs, mu, Nb)
        g = multiple_pdfs(self.obs, mu, N)   
        
        coef = (1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det(self.obsvar))))
        
        var = coef*f - g**2
        C = -1*var 

        return C
    
    def make_int(self, bdg, n_frac):
        # Make integer
        n_floor = np.floor(n_frac)
        remain = n_frac - n_floor
        tot_remain = bdg - np.sum(n_floor)
        sorted_indices = np.array(np.argsort(remain)[::-1])
        if tot_remain > 0:
            idx = sorted_indices[np.arange(0, tot_remain, dtype=int)]
            n_floor[idx] = n_floor[idx] + 1
        
        return n_floor
    
    def ivar_exploit(self, emu, pc_settings):
        texploit = self.theta
        rep = self.reps
        x = self.x
        tmesh = self.theta_mesh
        obs = self.obs
        obsvar = self.obsvar

        fE = emu._info['f']
        tE = emu._info['theta']
        for cid, ct in enumerate(texploit):
            if rep[cid] > 0:
                fE, tE = impute(ct=ct[None, :], x=x, fE=fE, tE=tE, reps=rep[cid], emu=emu, rnd_str=self.rand_stream)

        emu = build_emulator(x=x, theta=tE, f=fE, pcset=pc_settings)
        ivar = compute_ivar(emu=emu, ttest=tmesh, x=x, obs=obs, obsvar=obsvar)    
        
        return ivar
        
        