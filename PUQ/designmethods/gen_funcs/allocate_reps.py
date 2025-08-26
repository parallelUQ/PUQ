import numpy as np
import scipy
from numpy.linalg import cholesky, inv, det
from hetgpy.covariance_functions import cov_gen
from PUQ.designmethods.support import multiple_pdfs

class allocate:
    def __init__(self, 
                 budget, 
                 model, 
                 func_cls, 
                 mesh, 
                 trace=1,
                 alloc_settings={},
                 rand_stream=None):
        
        self.budget = budget
        self.model = model
        self.x = func_cls.x
        self.theta_mesh = mesh["theta"]
        self.w_mesh = mesh["weight"]
        self.obs = func_cls.real_data
        self.obsvar = func_cls.obsvar
        self.use_Ki = alloc_settings.get('use_Ki', True) 
        self.eps = np.sqrt(np.finfo(float).eps)
        self.trace = trace
        self.func_cls = func_cls
        self.rand_stream = rand_stream
        self.z = self.model['X0']
        self.a0 = self.model['mult']
     
        

    def allocatereps(self):
        
        Cinfo = self.compute_ivar_weights()
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

         
    def compute_ivar_weights(self):
    
        d = len(self.x)
        n_integ = self.theta_mesh.shape[0]
        n = self.z.shape[0]

        #pred_nugs = self.emu.predict(x=self.x, theta=self.theta, thetaprime=None)
        #pred_mesh = self.model.predict(x=self.theta_mesh)
        
        # obsvar3d = self.obsvar.reshape(1, d, d) 
        # # ntest x d
        # mu = pred_mesh._info['mean'].T 
        # S = pred_mesh._info['S'] 
        # St = np.transpose(S, (2, 0, 1))
        
        obsvar3d = self.obsvar.reshape(1, d, d)
        
        x = self.x
        ntot = n_integ*d
        t = self.theta_mesh
        x_tiled = np.tile(x, (t.shape[0], 1))
        t_repeated = np.repeat(t, x.shape[0], axis=0)
        z = np.hstack([x_tiled, t_repeated])

        # to construct S matrix
        id_row = np.arange(0, ntot)
        id_col = np.arange(0, ntot).reshape(n_integ, d)
        id_col = np.repeat(id_col, repeats=d, axis=0)

         # predict at mesh       
        meshPr = self.model.predict(x=z, xprime=z)
        
        # ntot, ntot x ntot, ntot
        mu, Sn = meshPr["mean"], meshPr["cov"]

        muT = mu.reshape(n_integ, d)
        S = Sn[id_row[:, None], id_col].reshape(n_integ, d, d)
        
        # ntest x d x d
        Nb = S + 0.5*obsvar3d
        N = S + obsvar3d

        f = multiple_pdfs(self.obs, muT, Nb)
        g = multiple_pdfs(self.obs, muT, N)
        
        # ntest x 1 x d
        h = (self.obs - muT).reshape(n_integ, 1, d)
    
        # ntest x d x d
        Nbinv = inv(Nb)

        # ntest x d x d
        Ninv = inv(N)

        # ntest x 1 x d
        hNb = np.matmul(h, Nbinv)
        hN = np.matmul(h, Ninv)

        coef = (1/((2**d)*(np.sqrt(np.pi)**d)*np.sqrt(det(self.obsvar))))
  
        Mb = np.zeros((n, n_integ, d, d), dtype=np.float64)

        pred_nugs = self.model.predict(x=self.z, nugs_only=True)

        # n x n
        if self.use_Ki:
            Ki = self.model['Ki']
        else:
            K = cov_gen(X1=self.z, theta=self.model['theta'], type = self.model['covtype'])
            Ki = scipy.linalg.pinv(K, rcond=self.eps)
                
        ids = np.arange(0, len(z), d)
        for i in range(0, n):
            J = np.zeros((n, n))
            J[i, i] = 1

            # n x n
            A = Ki@J@Ki

            for j in range(0, d):
                # n_integ x 1
                zo = z[ids + j]
                
                # n x n_integ
                kv = cov_gen(X1=self.z, X2=zo, theta=self.model['theta'], type = self.model['covtype'])

                for jp in range(0, d):
                    # n_integ x 1
                    zop = z[ids + jp]
                    
                    # n x n_integ
                    kvp = cov_gen(X1=self.z, X2=zop, theta=self.model['theta'], type = self.model['covtype'])

                    Mb[i, :, j, jp] = -pred_nugs["nugs"][i]*np.einsum('ji,jk,ki->i', kv, A, kvp) 


        dlogfdai = np.zeros((n, n_integ))
        dloggdai = np.zeros((n, n_integ))
        C = np.zeros((n, 1))
        for i in range(0, n):

            M = Mb[i, :, :, :]
  
            part1f = -0.5 * np.trace(np.matmul(Nbinv, M), axis1=1, axis2=2)
            part2f = 0.5 * (np.matmul(np.matmul(hNb, M), np.transpose(hNb, (0, 2, 1))))
            part1g = -0.5 * np.trace(np.matmul(Ninv, M), axis1=1, axis2=2)
            part2g = 0.5 * (np.matmul(np.matmul(hN, M), np.transpose(hN, (0, 2, 1))))
            
            dlogfdai = part1f + part2f.flatten()
            dloggdai = part1g + part2g.flatten()
            subC = self.w_mesh*(coef*(f*dlogfdai) - 2*(g**2)*dloggdai)
            
            C[i] = np.sum(subC)

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
    

        