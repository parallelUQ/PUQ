import numpy as np
from ptest_funcs import pritam, sinfunc
from sklearn.linear_model import LinearRegression


def find_bestfit_param(ex='sinfunc'):
    bias = True
    
    if ex == 'pritam':
        cls_data = pritam()
        
        x1 = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 20)
        x2 = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 20)
        X1, X2 = np.meshgrid(x1, x2)
        xmesh = np.vstack([X1.ravel(), X2.ravel()]).T
        ytest = cls_data.function(xmesh[:, 0], xmesh[:, 1], cls_data.true_theta).reshape(1, len(xmesh)) + cls_data.bias(xmesh[:, 0], xmesh[:, 1]).reshape(1, len(xmesh))
        f = cls_data.function(xmesh[:, 0], xmesh[:, 1], cls_data.true_theta).reshape(1, len(xmesh))
        
        cand_theta = np.linspace(cls_data.thetalimits[2][0], cls_data.thetalimits[2][1], 1000)
        fs = []
        for cand in cand_theta:
            feval = cls_data.function(xmesh[:, 0], xmesh[:, 1], np.array(cand)).reshape(1, len(xmesh))
            
            # Predict linear bias mean  
            bias = (ytest - feval).T
        
            model = LinearRegression().fit(xmesh, bias)
            bias_hat = model.predict(xmesh)
            diff = bias - bias_hat
                
            fs.append(np.sum((diff)**2))
        thetamle = cand_theta[np.argmin(fs)]

    elif ex == 'sinfunc':
        cls_data = sinfunc()
    
        # # # Create a mesh for test set # # # 
        xmesh = np.linspace(cls_data.thetalimits[0][0], cls_data.thetalimits[0][1], 1000)[:, None]
        ytest = (cls_data.function(xmesh, cls_data.true_theta) + cls_data.bias(xmesh)).T 
        f = cls_data.function(xmesh, cls_data.true_theta).reshape(1, len(xmesh))
        
        cand_theta = np.linspace(cls_data.thetalimits[1][0], cls_data.thetalimits[1][1], 1000)
        fs = []
        for cand in cand_theta:
            feval = cls_data.function(xmesh, np.array(cand)).reshape(1, len(xmesh))
            
            # Predict linear bias mean  
            bias = (ytest - feval).T
        
            model = LinearRegression().fit(xmesh, bias)
            bias_hat = model.predict(xmesh)
            diff = bias - bias_hat
                
            fs.append(np.sum((diff)**2))
        thetamle = cand_theta[np.argmin(fs)]
    
    return thetamle
