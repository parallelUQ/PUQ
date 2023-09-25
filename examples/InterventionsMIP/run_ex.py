#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:57:21 2023

@author: ozgesurer
"""

runfile('/Users/ozgesurer/Desktop/COVID_Staged_Alert/InterventionsMIP/main_deterministic.py', 
        'austin -f setup_data_Final_lsq.json -t tiers5_opt_Final.json -train_reps 1 -test_reps 1 -f_config austin_test_IHT.json -n_proc 1 -gt [-1,0,5,20,50] -tr transmission_Final_lsq.csv -hos austin_real_hosp_lsq.csv')