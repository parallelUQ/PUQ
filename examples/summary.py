from PUQ.utils import parse_arguments, read_output
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from test_funcs import bimodal, banana, unimodal
from utilities import test_data_gen, twoD
import scipy.stats as sps
    
def read_data(rep0=0, 
              repf=10, 
              methods=['ivar', 'imse', 'unif'], 
              batches=[8, 16, 32, 64], 
              examples=['unimodal', 'banana', 'bimodal'], 
              ids=['3', '1', '2'],
              ee='exploit',
              metric='TV', 
              folderpath=None, 
              ntotal=192,
              initial=30):
    
    datalist1, datalist2 = [], []
    for eid, example in enumerate(examples):
        for m in methods:
            for bid, b in enumerate(batches):
                path = folderpath + ids[eid] + '_' + example + '_' + ee + '/'  + str(b) + '/'
                for r in range(rep0[eid], repf[eid]):
                    desobj = read_output(path, example, m, b+1, b, r)
                    reps0 = desobj._info['reps0']
                    theta0 = desobj._info['theta0']
                    
                    f = desobj._info['f']

                    if np.unique(f, axis=1).shape[1] != (ntotal[eid] + initial[eid]):
                        print(np.unique(f, axis=1).shape)
                    
                    if metric == 'TV':
                        for tvid, tv in enumerate(desobj._info['TV']):
                            datalist1.append({'MAD':tv, 't':tvid, 'rep':r, 'batch':b, 'worker':b+1, 'method':m, 'example':example})
                    else:
                        for tvid, tv in enumerate(desobj._info['TViter']):
                            if tvid < ntotal[eid]+1:
                                datalist1.append({'MAD':tv, 't':tvid, 'rep':r, 'batch':b, 'worker':b+1, 'method':m, 'example':example})
                            
                    datalist2.append({'rep':r, 'batch':b, 'worker':b+1, 'method':m, 'example':example,
                                     'iter_explore':desobj._info['iter_explore'], 'iter_exploit':desobj._info['iter_exploit']})
    
    df1 = pd.DataFrame(datalist1)
    df2 = pd.DataFrame(datalist2)
    return df1, df2


def boxplot(df, examples, batches):
    for example in examples:
        for b in batches:
            df1 = df.loc[df['example'] == example]
            df2 = df1.loc[df1['batch'] == b]
            sns.boxplot(x = df2['t'], 
                        y = df2['MAD'], 
                        hue = df2['method'], 
                        showfliers=False,
                        palette = 'Set2').set_title(example + '_' + str(b))
            plt.show()
            
def boxplot_batch(df, examples, methods):
    for example in examples:
        for m in methods:
            df1 = df.loc[df['example'] == example]
            df2 = df1.loc[df['method'] == m]
            fig, ax = plt.subplots()
            #fig.set_size_inches(20, 5)
            values_list = [0, 64, 128, 192, 256]
            dfnew = df2[df2['t'].isin(values_list)]
            sns.boxplot(x = dfnew['t'], 
                        y = dfnew['MAD'], 
                        hue = dfnew['batch'], 
                        showfliers=False,
                        palette = 'Set2').set_title(example + '_' + m)
            plt.show()

def lineplot(df, examples, batches, metric='TV', ci=None, save=False):
    if len(examples) == 3:
        ft = 20
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        for i, example in enumerate(examples):
            df1 = df.loc[df['example'] == example]
            sns.lineplot(data=df1, x="t", y='MAD', hue='method', style='batch', palette=['r', 'g', 'b'], ci=ci, linewidth=5, ax=ax[i])
            if example == "bimodal":
                lgd = ax[i].legend(loc = 'upper center', bbox_to_anchor = (1.2, 0.8),
                          fancybox = True, shadow = True, ncol = 1, fontsize=ft-5)
            else:
                ax[i].legend([],[], frameon=False)
            ax[i].set_yscale('log')
            if metric == 'TV':
                ax[i].set_xlabel('t', fontsize=ft)
            else:
                ax[i].set_xlabel('# of simulation evaluations', fontsize=ft)
            ax[i].set_ylabel('MAD', fontsize=ft)   
            ax[i].tick_params(axis="both", labelsize=ft)
        plt.show()
    elif len(examples) == 2:
        ft = 20
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        for i, example in enumerate(examples):
            df1 = df.loc[df['example'] == example]
            sns.lineplot(data=df1, x="t", y='MAD', hue='method', style='batch', palette=['r', 'g', 'b'], ci=ci, linewidth=5, ax=ax[i])
            if example == "SEIRDS":
                lgd = ax[i].legend(loc = 'upper center', bbox_to_anchor = (1.2, 0.8),
                          fancybox = True, shadow = True, ncol = 1, fontsize=ft-5)
            else:
                ax[i].legend([],[], frameon=False)
            ax[i].set_yscale('log')
            if metric == 'TV':
                ax[i].set_xlabel('t', fontsize=ft)
            else:
                ax[i].set_xlabel('# of simulation evaluations', fontsize=ft)
            ax[i].set_ylabel('MAD', fontsize=ft)   
            ax[i].tick_params(axis="both", labelsize=ft)
        plt.show()
    else:
        ft = 20
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.lineplot(data=df, x="t", y='MAD', hue='method', style='batch', palette=['r', 'g', 'b', 'c'], ci=ci, linewidth=5, ax=ax)
        lgd = ax.legend(loc = 'upper center', bbox_to_anchor = (1.2, 0.8),
                  fancybox = True, shadow = True, ncol = 1, fontsize=ft-5)
        ax.set_yscale('log')
        if metric == 'TV':
            ax.set_xlabel('t', fontsize=ft)
        else:
            ax.set_xlabel('# of simulation evaluations', fontsize=ft)
        ax.set_ylabel('MAD', fontsize=ft)   
        ax.tick_params(axis="both", labelsize=ft)
        if save:
            plt.savefig("Figure4" + ".png", bbox_inches="tight")
        plt.show()

def exp_ratio(dfexpl, examples, methods, batches, ntotals):

    ft = 20
    for mid, m in enumerate(methods):
        print(m)
        vals, vals2, vals3 = [], [], []
        for i, ex in enumerate(examples):
            ntotal = ntotals[i]
            for bid, b in enumerate(batches):
                dffilt = dfexpl.loc[(dfexpl['method'] == m) & (dfexpl['example'] == ex) & (dfexpl['batch'] == b)]
                # vals.append({"b":b, 
                #              "example":ex, 
                #              "exploration":np.mean(dffilt['iter_explore'])/(ntotal/b), 
                #              "exploitation":np.mean(dffilt['iter_exploit'])/(ntotal/b)})
                
                vals2.extend({"b":b, 
                             "example":ex, 
                             "exploration":e} for e in dffilt['iter_explore']/(ntotal/b))
                
                # vals3.extend({"b":b, 
                #              "example":ex, 
                #              "exploitation":e} for e in dffilt['iter_exploit']/(ntotal/b))
                
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        dfnew2 = pd.DataFrame(vals2)
        dfnew2['percent'] = dfnew2['exploration']*100
        # print(dfnew2)
        sns.boxplot(x='example', y='percent', hue='b', data=dfnew2, ax=ax, showfliers=False)
        ax.set_ylabel('Exploration stages (%)', fontsize=ft)  
        ax.set_xlabel('Example', fontsize=ft)  
        ax.tick_params(axis="both", labelsize=ft)
        lgd = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3),
                  fancybox = True, shadow = True, ncol = 4, fontsize=ft-5)
        ax.set_ylim(0, 101)
        plt.axhline(y=50, color='red', ls='--', lw=5)
        plt.show()
        
        # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # dfnew2 = pd.DataFrame(vals3)
        # dfnew2['percent'] = dfnew2['exploitation']*100
        # sns.boxplot(x='example', y='percent', hue='b', data=dfnew2, ax=ax)
        # ax.set_ylabel('Exploration stages (%)', fontsize=ft)  
        # ax.set_xlabel('Example', fontsize=ft)  
        # ax.tick_params(axis="both", labelsize=ft)
        # lgd = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3),
        #           fancybox = True, shadow = True, ncol = 4, fontsize=ft-5)
        # ax.set_ylim(0, 101)
        # plt.axhline(y=50, color='red', ls='--', lw=5)
        # plt.show()
        

def visual_theta(example, method, batch, rep0, repf, initial, folderpath, ids, ee):

    path = folderpath + ids + '_' + example + '_' + ee + '/'  + str(batch) + '/'
    for r in range(rep0, repf):
        desobj = read_output(path, example, method, batch+1, batch, r)
        reps0 = desobj._info['reps0']
        theta0 = desobj._info['theta0']
        
        f = desobj._info['f']
        theta = desobj._info['theta']
        print(np.unique(f, axis=1).shape)
        print(np.unique(theta, axis=0).shape)
        
        nmesh = 50
        cls_func = eval(example)()
        cls_func.realdata(seed=r)

        theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
        twoD(desobj, Xpl, Ypl, p_test, nmesh)


def interval_score(examples, methods, batches, rep0, repf, initial, ids, folderpath, ee):
    
    for exid, ex in enumerate(examples):
        result = []
        for r in range(rep0[exid], repf[exid]):
            cls_func = eval(ex)()
            cls_func.realdata(seed=r)
            nmesh = 50
            theta_test, p_test, f_test, Xpl, Ypl = test_data_gen(cls_func, nmesh)
            thetamle = theta_test[np.argmax(p_test), :]
            #print(thetamle)
            for mid, m in enumerate(methods):
                for bid, b in enumerate(batches):
                    path = folderpath + ids[exid] + '_' + ex + '_' + ee + '/'  + str(b) + '/'
                    desobj = read_output(path, ex, m, b+1, b, r)
                    theta = desobj._info['theta'][initial[exid]:, :]
                    for i in range(theta.shape[1]):
                        total_is = compute_interval_score(theta[:, i], thetamle[i])
                        result.append({'i': i, 'score': total_is, 'method': m, 'batch': b, 'example': ex})
        
        print(ex)
        df = pd.DataFrame(result)
        for mid, m in enumerate(methods):
            print(m)
            for i in range(theta.shape[1]):
                print('\u03B8', end=', ')
                for bid, b in enumerate(batches):
                    df1 = df.loc[df['method'] == m]
                    df2 = df1.loc[df['batch'] == b]
                    df3 = df2.loc[df['i'] == i] 
                    print(np.round(np.median(df3['score']), 1), end=' ')
                    print('(' + str(np.round(np.std(df3['score']), 1)) + ')', end=', ')       
                print()
                
        # medians = df.groupby(['i', 'method', 'batch']).median().round(1)
        # sds = df.groupby(['i', 'method', 'batch']).std().round(1)
        # print('median')
        # print(medians)
        # print(df.groupby(['i', 'method', 'batch']).median().round(1))
        # print('std')
        # print(df.groupby(['i', 'method', 'batch']).std().round(1))
        # for i in range(theta.shape[1]):
        #     df1 = df.loc[df['i'] == i]
        #     fig, ax = plt.subplots()
        #     sns.boxplot(x = df1['batch'], 
        #                 y = df1['score'], 
        #                 hue = df1['method'], 
        #                 showfliers=True,
        #                 palette = 'Set2')
        #     plt.show()

        
def compute_interval_score(theta, thetamle):
    alpha = 0.1
    u = np.quantile(theta, 1-alpha/2)
    l = np.quantile(theta, alpha/2)

    is_l = 1 if thetamle < l else 0
    is_u = 1 if thetamle > u else 0

    total_is = (u - l) + (2/alpha) * (l-thetamle) * (is_l) + (2/alpha) * (thetamle-u) * (is_u)

    return total_is        

def interval_score_SIR(examples, methods, batches, rep0, repf, initial, ids, folderpath, ee):
    from smt.sampling_methods import LHS
    for exid, ex in enumerate(examples):
        
        # # Create test data
        n0 = initial[exid]
        nmesh = 50
        nt = nmesh**2
        nrep = 1000
        cls_func = eval(ex)()
        if ex == 'SIR':
            xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
            ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
            Xpl, Ypl = np.meshgrid(xpl, ypl)
            theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T
        elif ex == 'SEIRDS':
            sampling = LHS(xlimits=cls_func.thetalimits, random_state=100)
            theta_test = sampling(nt)
            
        f_test = np.zeros((nt, cls_func.d))
        persis_info = {'rand_stream': np.random.default_rng(100)}
        for thid, th in enumerate(theta_test):
            IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
            f_test[thid, :] = np.mean(IrIdRD, axis=0)

        result = []
        for r in range(rep0[exid], repf[exid]):
            cls_func = eval(ex)()
            cls_func.realdata(seed=r)
            p_test = np.zeros(nmesh**2)
            for thid, th in enumerate(theta_test):
                rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
                p_test[thid] = rnd.pdf(cls_func.real_data)
            thetamle = theta_test[np.argmax(p_test), :]
            for mid, m in enumerate(methods):
                for bid, b in enumerate(batches):
                    path = folderpath + ids[exid] + '_' + ex + '_' + ee + '/'  + str(b) + '/'
                    desobj = read_output(path, ex, m, b+1, b, r)
                    theta = desobj._info['theta'][n0:, :]
                    for i in range(theta.shape[1]):
                        total_is = compute_interval_score(theta[:, i], thetamle[i])
                        result.append({'i': i, 'score': total_is, 'method': m, 'batch': b, 'example': ex})
        
        df = pd.DataFrame(result)
        for mid, m in enumerate(methods):
            print(m)
            for i in range(theta.shape[1]):
                print('\u03B8', end=', ')
                for bid, b in enumerate(batches):
                    df1 = df.loc[df['method'] == m]
                    df2 = df1.loc[df['batch'] == b]
                    df3 = df2.loc[df['i'] == i] 
                    print(np.round(np.median(df3['score']), 1), end=' ')
                    print('(' + str(np.round(np.std(df3['score']), 1)) + ')', end=', ')       
                print()

        
def SIR2D(example, batch, method, r, ids=None, ee=None, folderpath=None):

    from matplotlib.colors import ListedColormap
    yellow_colors = [
        (1, 1, 1),
        (1, 1, 0.8),  # light yellow
        (1, 1, 0.6),  
        (1, 1, 0.4),  
        (1, 1, 0.2),  
        (1, 1, 0),    # yellow    
        (1, 0.9, 0),  # dark yellow
        (1, 0.8, 0),  # yellow-orange
        (1, 0.6, 0),  # orange
        (1, 0.4, 0),  # dark orange
        (1, 0.2, 0)   # very dark orange
    ]
    yellow_cmap = ListedColormap(yellow_colors, name='yellow')
    
    path = folderpath + ids + '_' + example + '_' + ee + '/'  + str(batch) + '/'
    desobj = read_output(path, example, method, batch+1, batch, r)
    theta0 = desobj._info['theta0']
    reps0 = desobj._info['reps0']   
    
    print("Iter explore")
    print(desobj._info['iter_explore'])
    
    print("Iter exploit")
    print(desobj._info['iter_exploit'])
    
    nmesh = 50
    nt = nmesh**2
    # Create test data
    nrep = 1000
    persis_info = {'rand_stream': np.random.default_rng(100)}
    cls_func = eval('SIR')()
    cls_func.realdata(seed=r)
    
    from smt.sampling_methods import LHS
    n0 = 15
    sampling = LHS(xlimits=cls_func.thetalimits, random_state=int(r))
    thetainit = sampling(n0)

    xpl = np.linspace(cls_func.thetalimits[0][0], cls_func.thetalimits[0][1], nmesh)
    ypl = np.linspace(cls_func.thetalimits[1][0], cls_func.thetalimits[1][1], nmesh)
    Xpl, Ypl = np.meshgrid(xpl, ypl)
    theta_test = np.vstack([Xpl.ravel(), Ypl.ravel()]).T

    f_test = np.zeros((nt, cls_func.d))
    f_var = np.zeros((nt, cls_func.d))
    for thid, th in enumerate(theta_test):
        IrIdRD          = cls_func.sim_f(thetas=th, return_all=True, repl=nrep, persis_info=persis_info)
        f_test[thid, :] = np.mean(IrIdRD, axis=0)
        f_var[thid, :]  = np.var(IrIdRD, axis=0)

        
    for i in range(0, 3):
        fig, ax = plt.subplots()
        cs = ax.contourf(Xpl, Ypl, f_var[:, i].reshape(nmesh, nmesh), cmap=yellow_cmap, alpha=0.75)
        cbar = fig.colorbar(cs)
        CS = ax.contour(Xpl, Ypl, f_test[:, i].reshape(nmesh, nmesh), colors='black')
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_xlabel(r"$\theta_1$", fontsize=16)
        ax.set_ylabel(r"$\theta_2$", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        plt.show() 

    p_test = np.zeros(nmesh**2)
    for thid, th in enumerate(theta_test):
        rnd = sps.multivariate_normal(mean=f_test[thid, :], cov=cls_func.obsvar)
        p_test[thid] = rnd.pdf(cls_func.real_data)
            


    fig, ax = plt.subplots()
    cs = ax.contourf(Xpl, Ypl, np.sum(f_var, axis=1).reshape(nmesh, nmesh), cmap=yellow_cmap, alpha=0.75)
    cbar = fig.colorbar(cs)
    cp = ax.contour(Xpl, Ypl, p_test.reshape(nmesh, nmesh), 20, cmap="coolwarm")
    for label, x_count, y_count in zip(reps0, theta0[:, 0], theta0[:, 1]):
        if np.array([x_count, y_count]) in thetainit:
            plt.annotate(label, xy=(x_count, y_count), xytext=(0, 0), textcoords='offset points', fontsize=12, color='cyan')  
        else:
            plt.annotate(label, xy=(x_count, y_count), xytext=(0, 0), textcoords='offset points', fontsize=12, color='black')  

    ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    plt.show() 
    
def SIRtheta(example, batch, method, r, initial, ids=None, ee=None, folderpath=None):
    cls_func = eval(str(example))()
    p = cls_func.p
    path = folderpath + ids + '_' + example + '_' + ee + '/'  + str(batch) + '/'
    
    desobj = read_output(path, example, method, batch+1, batch, r)
    print(desobj._info['iter_explore'])
    print(desobj._info['iter_exploit'])
    thetas = desobj._info['theta']
    pdtheta = pd.DataFrame(thetas) 

    pdtheta['color'] = np.concatenate((np.repeat('red', initial), np.repeat('gray', 256)))
    labs = [r'$\theta_1$', 
            r'$\theta_2$', 
            r'$\theta_3$', 
            r'$\theta_4$', 
            r'$\theta_5$', 
            r'$\theta_6$', 
            r'$\theta_7$']
    sns.set_theme(style='white')
    g = sns.pairplot(pdtheta, 
                     kind='scatter',
                     diag_kind='hist',
                     corner=True,
                     hue="color",
                     palette=['blue', 'gray'],
                     markers=["*", "X"])

    
    ft = 20
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import FormatStrFormatter
    for i in range(0, p):
        for j in range(0, i+1):
            g.axes[i, j].set_xlim((0, 1))
    
    for j in range(0, p):
        g.axes[j, j].axvline(x=cls_func.theta_true[j], color='red', linestyle='--', lw=2)

    for j in range(0, p):
        lab = "\theta_" + str(j)
        g.axes[j, 0].set_ylabel(labs[j], fontsize=ft)
        g.axes[p-1, j].set_xlabel(labs[j], fontsize=ft)

    for j in range(1, p):    
        g.axes[j, 0].yaxis.set_major_locator(MaxNLocator(3))
        g.axes[j, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        g.axes[j, 0].tick_params(axis='both', which='major', labelsize=ft-2)
    
    for j in range(0, p):    
        g.axes[p-1, j].xaxis.set_major_locator(MaxNLocator(3))
        g.axes[p-1, j].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        g.axes[p-1, j].tick_params(axis='both', which='major', labelsize=ft-2)

    g._legend.remove()
    #plt.savefig("Figure13_" + method + ".png", bbox_inches="tight")
    plt.show()
    return 

def SIRfuncevals(example, method, batch, r, ids, ee, initial, folderpath):
    
    ft = 20
    path = folderpath + ids + '_' + example + '_' + ee + '/'  + str(batch) + '/'
    desobj = read_output(path, example, method, batch+1, batch, r)
    thetas = desobj._info['theta'][initial:, ]
    cls_func = eval(example)()
    persis_info = {'rand_stream': np.random.default_rng(100)}

    if example == 'SIR':
        fig, axs = plt.subplots(1, 3, figsize=(25, 6))
    else:
        fig, axs = plt.subplots(2, 3, figsize=(25, 12))
        
    for th in thetas:
        if example == 'SIR':
            S, I, R = cls_func.simulation(thetas=th, repl=1000, persis_info=persis_info)
        else:
            S, E, Ir, Id, R, D = cls_func.simulation(thetas=th, repl=1000, persis_info=persis_info)           

        if example == 'SIR':
            axs[0].plot(np.mean(S, axis=1), c='orange', alpha=0.5)
            axs[1].plot(np.mean(I, axis=1), c='pink', alpha=0.5)
            axs[2].plot(np.mean(R, axis=1), c='cyan', alpha=0.5)
        else:
            axs[0, 0].plot(np.mean(S, axis=1), c='orange', alpha=0.3)
            axs[0, 1].plot(np.mean(E, axis=1), c='pink', alpha=0.3)
            axs[0, 2].plot(np.mean(Ir, axis=1), c='cyan', alpha=0.3)
            axs[1, 0].plot(np.mean(Id, axis=1), c='violet', alpha=0.3)
            axs[1, 1].plot(np.mean(R, axis=1), c='yellow', alpha=0.3)
            axs[1, 2].plot(np.mean(D, axis=1), c='lime', alpha=0.3)
        
    if example == 'SIR':
        Strue, Itrue, Rtrue = cls_func.simulation(thetas=cls_func.theta_true, repl=1000, persis_info=persis_info) #cls_func.simulation(thetas=cls_func.theta_true, repl=100, **kwargs)
        axs[0].set_ylabel("Susceptible Individuals", fontsize=ft)
        axs[0].set_xlabel("Time", fontsize=ft)
        axs[0].tick_params(axis="both", labelsize=ft-2)
        axs[1].set_ylabel("Infected Individuals", fontsize=ft)
        axs[1].set_xlabel("Time", fontsize=ft)
        axs[1].tick_params(axis="both", labelsize=ft-2)
        axs[2].set_ylabel("Recovered Individuals", fontsize=ft)
        axs[2].set_xlabel("Time", fontsize=ft)
        axs[2].tick_params(axis="both", labelsize=ft-2)
    else:
        Strue, Etrue, Irtrue, Idtrue, Rtrue, Dtrue = cls_func.simulation(thetas=cls_func.theta_true, repl=1000, persis_info=persis_info)
        axs[0, 0].set_ylabel("Susceptible", fontsize=ft)
        axs[0, 0].set_xlabel("Time", fontsize=ft)
        axs[0, 0].tick_params(axis="both", labelsize=ft-2)
        axs[0, 1].set_ylabel("Exposed", fontsize=ft)
        axs[0, 1].set_xlabel("Time", fontsize=ft)
        axs[0, 1].tick_params(axis="both", labelsize=ft-2)
        axs[0, 2].set_ylabel("Infected (Recover)", fontsize=ft)
        axs[0, 2].set_xlabel("Time", fontsize=ft)
        axs[0, 2].tick_params(axis="both", labelsize=ft-2)
        axs[1, 0].set_ylabel("Infected (Dead)", fontsize=ft)
        axs[1, 0].set_xlabel("Time", fontsize=ft)
        axs[1, 0].tick_params(axis="both", labelsize=ft-2)
        axs[1, 1].set_ylabel("Recovered", fontsize=ft)
        axs[1, 1].set_xlabel("Time", fontsize=ft)
        axs[1, 1].tick_params(axis="both", labelsize=ft-2)
        axs[1, 2].set_ylabel("Dead", fontsize=ft)
        axs[1, 2].set_xlabel("Time", fontsize=ft)
        axs[1, 2].tick_params(axis="both", labelsize=ft-2)

    if example == 'SIR':
        axs[0].plot(np.mean(Strue, axis=1), c='black', linestyle='dotted', linewidth=5)
        axs[1].plot(np.mean(Itrue, axis=1), c='black', linestyle='dotted', linewidth=5)
        axs[2].plot(np.mean(Rtrue, axis=1), c='black', linestyle='dotted', linewidth=5)
    else:
        axs[0, 0].plot(np.mean(Strue, axis=1), c='black', linestyle='dotted', linewidth=5)
        axs[0, 1].plot(np.mean(Etrue, axis=1), c='black', linestyle='dotted', linewidth=5)
        axs[0, 2].plot(np.mean(Irtrue, axis=1), c='black', linestyle='dotted', linewidth=5)
        axs[1, 0].plot(np.mean(Idtrue, axis=1), c='black', linestyle='dotted', linewidth=5)
        axs[1, 1].plot(np.mean(Rtrue, axis=1), c='black', linestyle='dotted', linewidth=5)
        axs[1, 2].plot(np.mean(Dtrue, axis=1), c='black', linestyle='dotted', linewidth=5)        
    plt.show()