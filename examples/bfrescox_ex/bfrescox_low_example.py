import os
import seaborn as sns
import pandas as pd
import scipy.stats as sps
from generate_test_data import generate_test_data
import numpy as np
import matplotlib.pyplot as plt
from PUQ.design import designer
from PUQ.designmethods.utils import parse_arguments, save_output


class bfrescox:
    def __init__(self):
        self.data_name = 'bfrescox'
        self.thetalimits = np.array([[40, 60], # V
                                     [0.7, 1.2], # r
                                     # [0.5, 0.8], # a
                                     [2.5, 4.5]]) # Ws
                                     #[0.5, 1.5],
                                     #[0.1, 0.4]])

        self.d = 15
        self.p = 3
        self.x = np.arange(0, self.d)[:, None]
        self.real_x = np.arange(0, self.d)[:, None] #np.array([[26, 31, 41, 51, 61, 71, 76, 81, 91, 101, 111, 121, 131, 141, 151]]).T
        self.real_data = np.log(np.array([[1243, 887.7, 355.5, 111.5, 26.5, 10.4, 8.3, 
                                           7.3, 17.2, 37.6, 48.7, 38.9, 32.4, 36.4, 61.9]], dtype='float64'))
        self.obsvar = np.diag(np.repeat(0.1, 15))
        self.out = [('f', float, (self.d,))]

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
        f = f[np.array([[26, 31, 41, 51, 61, 71, 76, 81, 91, 101, 111, 121, 131, 141, 151]])]
        return f
    
    def sim(self, H, persis_info, sim_specs, libE_info):
        """
        Wraps frescox function
        """
        function = sim_specs['user']['function']
        H_o = np.zeros(1, dtype=sim_specs['out'])

        V = H['thetas'][0][0]
        r = H['thetas'][0][1]
        Ws = H['thetas'][0][2]
        
        # V = 49.2849
        # r = 0.9070
        a = 0.6798
        # Ws = 3.3944
        rs = 1.0941
        a2 = 0.2763

        parameter = [V, r, a, Ws, rs, a2]
        self.generate_input_file(parameter)
        H_o['f'] = function()
        for fname in os.listdir():
            if fname.startswith("fort"):
                os.remove(fname)
        return H_o, persis_info

args        = parse_arguments()
cls_fresco  = bfrescox()
test_data   = generate_test_data(cls_fresco)

al_fresco = designer(data_cls=cls_fresco, 
                 method='SEQCAL', 
                 args={'mini_batch': args.minibatch, 
                       'n_init_thetas': 32,
                       'nworkers': args.nworkers,
                       'AL': args.al_func,
                       'seed_n0': args.seed_n0,
                       'prior': 'uniform',
                       'data_test': test_data,
                       'max_evals': 132})

save_output(al_fresco, cls_fresco.data_name, args.al_func, args.nworkers, args.minibatch, args.seed_n0)

show = False
if show:
    theta_al = al_fresco._info['theta']
    TV       = al_fresco._info['TV']
    HD       = al_fresco._info['HD']
    
    sns.pairplot(pd.DataFrame(theta_al))
    plt.show()
    plt.scatter(np.arange(len(TV[32:])), TV[32:])
    plt.yscale('log')
    plt.ylabel('MAD')
    plt.show()

    map_parameters = [49.2849, 0.9070, 3.3944]
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 20}
    
    plt.rc('font', **font) 
    for i in range(3):
        for j in range(i+1):
            if i == j:
                ax[i, i].hist(theta_al[:, i], color='gray')
                ax[i, i].axvline(x=map_parameters[i], color='red')
            else:
                ax[i, j].scatter(theta_al[32:, j], theta_al[32:, i], color='gray')
                ax[i, j].scatter(theta_al[0:32, j], theta_al[0:32, i], color='blue')
    ax[1,0].set_ylabel(r'$\theta_2$', fontsize=16)
    ax[2,0].set_ylabel(r'$\theta_3$', fontsize=16)
    ax[2,0].set_xlabel(r'$\theta_1$', fontsize=16)
    ax[2,1].set_xlabel(r'$\theta_2$', fontsize=16)
    ax[2,2].set_xlabel(r'$\theta_3$', fontsize=16)
    ax[1,2].axis('off')
    ax[0,1].axis('off')
    ax[0,2].axis('off')
    plt.show()
    
    real_x = np.array([[26, 31, 41, 51, 61, 71, 76, 81, 91, 101, 111, 121, 131, 141, 151]]).T
    real_data = np.array([[1243, 887.7, 355.5, 111.5, 26.5, 10.4, 8.3, 7.3, 17.2, 37.6, 48.7, 38.9, 32.4, 36.4, 61.9]], dtype='float64')
    n0 = 32
    f = al_fresco._info['f']
    for i in range(theta_al.shape[0]):
        if i < n0:
            #if i > 0:
            plt.plot(np.arange(15), np.exp(f[i, :]), color='gray', alpha=1, zorder=1)
            #else:
            #    plt.plot(np.arange(15), np.exp(f[i, :]), color='gray', alpha=1, zorder=1, label='Initial sample')
        else:
            #if i < 231:
            plt.plot(np.arange(15), np.exp(f[i, :]), color='red', alpha=0.1, zorder=1)
            #else:
            #    plt.plot(np.arange(15), np.exp(f[i, :]), color='red', alpha=0.1, zorder=1, label='Acquired sample')
    plt.scatter(np.arange(15), real_data.T, color='black', marker='P', zorder=2, label='Data')
    plt.xticks(ticks=np.arange(15)[::3], labels=real_x.flatten()[::3], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Degree', fontsize=16)
    plt.ylabel('Cross section', fontsize=16)
    #plt.legend(fontsize=14)
    plt.yscale('log')
    plt.show()
    
