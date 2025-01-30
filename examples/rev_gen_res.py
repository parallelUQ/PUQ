#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:05:37 2025

@author: surero
"""



import os
cd = os.getcwd() 
    
Figure6 = True
Table2 = False
Figure8 = False
exploresynth = True

if exploresynth:
    from summary import read_data, lineplot, exp_ratio, interval_score
    # # # Observe boxplots
    examples = ['unimodal', 'banana', 'bimodal']
    ids = ['3', '1', '2']

    ntotal = [256, 256, 256]
    n0 = [30, 30, 30]
    batches = [8, 16, 32, 64]
    methods = ['ivar', 'imse', 'unif', 'var']

    df, dfexpl = read_data(rep0=[0, 0, 0], 
                            repf=[30, 30, 30], 
                            methods=methods, 
                            batches=batches, 
                            examples=examples,
                            ids=ids,
                            ee='explore',
                            folderpath=cd + "/",
                            ntotal=ntotal,
                            initial=n0)

    if Figure6:
        lineplot(df, examples, batches, label="Figure6a_rev.png")

    
    if Figure8:
        exp_ratio(dfexpl, examples, methods, batches, ntotals=ntotal)
            
    if Table2:
        interval_score(examples=examples, 
                        methods=methods,
                        batches=batches, 
                        rep0=[0, 0, 0], 
                        repf=[30, 30, 30], 
                        initial=n0, 
                        ids=ids, 
                        ee='explore',
                        folderpath=cd + "/") 
    if Figure6:   
        df, dfexpl = read_data(rep0=[0, 0, 0], 
                                repf=[30, 30, 30], 
                                methods=methods, 
                                batches=batches, 
                                examples=examples, 
                                ids=ids,
                                ee='explore',
                                metric='iter',
                                folderpath=cd + "/",
                                ntotal=ntotal,
                                initial=n0)
        
        lineplot(df, examples, batches, metric='iter', label="Figure6b_rev.png")
        
Figure10, Figure11, Figure12, Figure13, FigureE3, Table4 = True, False, False, False, False, False
epimodel = True

if epimodel: 

        from summary import read_data, lineplot, exp_ratio, SIR2D, SIRfuncevals, SEIRDSfuncevals, interval_score_SIR
    
        
        examples = ['SIR', 'SEIRDS']
        batches = [8, 16, 32, 64]
        methods = ['ivar', 'imse', 'unif', 'var']
        ids = ['4', '5']
        ee = 'explore'
        ntotal = [256, 384]
        n0 = [30, 100]
        
        df, dfexpl = read_data(rep0=[0, 0], 
                               repf=[30, 30], 
                               methods=methods, 
                               batches=batches, 
                               examples=examples,
                               ids=ids,
                               ee=ee,
                               folderpath=cd + "/",
                               ntotal=ntotal,
                               initial=n0)
        
        if Figure10:
            lineplot(df, examples, batches, label="Figure10a_rev.png")
        
        if Figure11:
            exp_ratio(dfexpl, examples, 
                      ['ivar', 'var', 'imse'], 
                      batches, 
                      ntotals=ntotal,
                      label="Figure11_rev.png")
        
    
        if Figure10:
            df, dfexpl = read_data(rep0=[0, 0], 
                                    repf=[30, 30], 
                                    methods=methods, 
                                    batches=batches, 
                                    examples=examples, 
                                    ids=ids,
                                    ee=ee,
                                    metric='iter',
                                    folderpath=cd + "/",
                                    ntotal=ntotal,
                                    initial=n0)
            
            lineplot(df, examples, batches, metric='iter', label="Figure10b_rev.png")
        
        if Figure12:
            SIR2D('SIR', 16, "var", 4, ids="4", ee="explore", folderpath=cd + "/")
        
        if Figure13:            
            # b = 16, r = 1
            SIRfuncevals(example='SIR', 
                          batch=16, 
                          r=1, 
                          ids='4', 
                          ee="explore", 
                          initial=30, 
                          folderpath=cd + "/")
        if FigureE3: 
            # # b = 16 r = 1
            SEIRDSfuncevals(example='SEIRDS', 
                          batch=16, 
                          r=1, 
                          ids='5', 
                          ee=ee, 
                          initial=100,
                          folderpath=cd + "/")
            
        if Table4:
            interval_score_SIR(examples=examples, 
                            methods=methods,
                            batches=batches, 
                            rep0=[0, 0], 
                            repf=[30, 30], 
                            initial=n0, 
                            ids=ids, 
                            ee='explore',
                            folderpath=cd + "/") 
