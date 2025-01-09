from summary import read_data, boxplot, lineplot, exp_ratio, SIR2D, SIRtheta, SIRfuncevals, visual_theta, boxplot_batch, interval_score, interval_score_SIR


synth = True
epi = False

if synth:
    # # # Observe boxplots
    examples = ['unimodal', 'banana', 'bimodal']
    ids = ['3', '1', '2']
    #examples = ['banana']
    #ids = ['1']
    ee = 'explore'
    if ee == 'exploit':
        ntotal = [192, 192, 192]
        n0 = [200, 200, 200]
    else:
        ntotal = [256, 256, 256]
        n0 = [30, 30, 30]

    batches = [8, 16, 32, 64]
    #batches = [64]
    methods = ['ivar', 'imse', 'unif']
    df, dfexpl = read_data(rep0=[0, 0, 0], 
                            repf=[30, 30, 30], 
                            methods=methods, 
                            batches=batches, 
                            examples=examples,
                            ids=ids,
                            ee=ee,
                            folderpath='/Users/surero/Desktop/stochastic/',#'/Users/ozgesurer/Desktop/stochastic/',
                            ntotal=ntotal,
                            initial=n0)
    
    
            
    # boxplot(df, examples, batches)
    lineplot(df, examples, batches, ci=None)
    
    interval_score(examples=examples, #examples,
                    methods=methods, #['ivar'], 
                    batches=batches, 
                    rep0=[0, 0, 0], 
                    repf=[30, 30, 30], 
                    initial=n0, 
                    ids=ids, 
                    ee=ee,
                    folderpath='/Users/surero/Desktop/stochastic/') # '/Users/ozgesurer/Desktop/stochastic/')
    
    if ee == 'explore':
        exp_ratio(dfexpl, examples, methods, batches, ntotals=ntotal)
    
    df, dfexpl = read_data(rep0=[0, 0, 0], 
                            repf=[30, 30, 30], 
                            methods=methods, 
                            batches=batches, 
                            examples=examples, 
                            ids=ids,
                            ee=ee,
                            metric='iter',
                            folderpath='/Users/surero/Desktop/stochastic/',#'/Users/ozgesurer/Desktop/stochastic/',
                            ntotal=ntotal,
                            initial=n0)
    
    lineplot(df, examples, batches, metric='iter', ci=None)
    # boxplot_batch(df, examples, methods)
    
    # for i in range(0, 10):
    #     visual_theta(example='bimodal',
    #                   method='ivar', 
    #                   batch=32, 
    #                   rep0=i, repf=i+1, 
    #                   initial=30, 
    #                   ids='2', 
    #                   ee='explore',
    #                   folderpath='/Users/surero/Desktop/stochastic/')




if epi:
    examples = ['SIR', 'SEIRDS']
    #examples = ['SEIRDS']
    batches = [8, 16, 32, 64]
    #batches = [16, 32, 64]
    methods = ['ivar', 'imse', 'unif']
    ids = ['4', '5']
    #ids = ['5']
    ee = 'explore'
    ntotal = [256, 384]
    #ntotal = [384]
    n0 = [30, 100]
    #n0 = [100]
    df, dfexpl = read_data(rep0=[0, 0], 
                           repf=[30, 10], 
                           methods=methods, 
                           batches=batches, 
                           examples=examples,
                           ids=ids,
                           ee=ee,
                           folderpath='/Users/surero/Desktop/stochastic/',
                           ntotal=ntotal,
                           initial=n0)
    
    # boxplot(df, examples, batches)
    lineplot(df, examples, batches, ci=None)
    exp_ratio(dfexpl, examples, methods, batches, ntotals=ntotal)
    
    # interval_score_SIR(examples=examples,
    #                     methods=methods, 
    #                     batches=batches, 
    #                     rep0=[0, 0], 
    #                     repf=[10, 10], 
    #                     initial=n0, 
    #                     ids=ids, 
    #                     folderpath='/Users/surero/Desktop/stochastic/',
    #                     ee=ee)
    
    df, dfexpl = read_data(rep0=[0, 0], 
                            repf=[30, 10], 
                            methods=methods, 
                            batches=batches, 
                            examples=examples, 
                            ids=ids,
                            ee=ee,
                            metric='iter',
                            folderpath='/Users/surero/Desktop/stochastic/',
                            ntotal=ntotal,
                            initial=n0)
    
    lineplot(df, examples, batches, metric='iter', ci=None)
    # boxplot_batch(df, examples, methods)


    

    r = 1
    b = 32
    # SIR2D(example='SIR', 
    #       batch=b, 
    #       method='ivar', 
    #       r=r, 
    #       ids='4', 
    #       ee=ee, 
    #       folderpath='/Users/ozgesurer/Desktop/stochastic/')
    # SIR2D(example='SIR', 
    #       batch=b, 
    #       method='imse', 
    #       r=r, 
    #       ids='4', 
    #       ee=ee, 
    #       folderpath='/Users/ozgesurer/Desktop/stochastic/')
    # for r in range(0, 10):
    #     SIRtheta(example='SEIRDS', method='ivar', batch=64, r=r, initial=100, ids=ids[0], ee=ee, folderpath='/Users/ozgesurer/Desktop/stochastic/')
    # SIRfuncevals(example='SIR', 
    #               method='ivar', 
    #               batch=b, 
    #               r=r, 
    #               ids='4', 
    #               ee=ee, 
    #               initial=30, 
    #               folderpath='/Users/ozgesurer/Desktop/stochastic/')
    # SIRfuncevals(example='SIR', 
    #               method='imse', 
    #               batch=b, 
    #               r=r, 
    #               ids='4', 
    #               ee=ee, 
    #               initial=30,
    #               folderpath='/Users/ozgesurer/Desktop/stochastic/')

    # r = 1
    # SIRfuncevals(example='SEIRDS', 
    #               method='ivar', 
    #               batch=b, 
    #               r=r, 
    #               ids='5', 
    #               ee=ee, 
    #               initial=100, 
    #               folderpath='/Users/ozgesurer/Desktop/stochastic/')
    # SIRfuncevals(example='SEIRDS', 
    #               method='imse', 
    #               batch=b, 
    #               r=r, 
    #               ids='5', 
    #               ee=ee, 
    #               initial=100, 
    #               folderpath='/Users/ozgesurer/Desktop/stochastic/')