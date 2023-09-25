if __name__ == '__main__':
    import multiprocessing as mp
    from utils import parse_arguments
    from InterventionsMIP import load_config_file, logger, change_paths
    
    # Parse arguments
    args = parse_arguments()
    # Load config file
    load_config_file(args.f_config)
    # Adjust paths
    change_paths(args)
    
    from instances import load_instance, load_tiers, load_seeds
    from downsampling import policy_search_filter
    from objective_functions import multi_tier_objective
    from policies import MultiTierPolicy as MTP
    
    # Parse city and get corresponding instance
    instance = load_instance(args.city, setup_file_name=args.f, transmission_file_name=args.tr, hospitalization_file_name=args.hos)
    train_seeds, test_seeds = load_seeds(args.city, args.seed)
    tiers = load_tiers(args.city, tier_file_name=args.t)
    
    # TODO Read command line args for n_proc for better integration with crunch
    n_proc = args.n_proc
    
    # TODO: pull out n_replicas_train and n_replicas_test to a config file
    n_replicas_train = args.train_reps
    n_replicas_test = args.test_reps
    
    # Create the pool (Note: pool needs to be created only once to run on a cluster)
    mp_pool = mp.Pool(n_proc) if n_proc > 1 else None
    
    # check if the "do-nothing" / 'Stage 1 option is in the tiers. If not, add it
    originInt = {
        "name": "Stage 1",
        "transmission_reduction": 0,
        "cocooning": 0,
        "school_closure": 0,
        "min_enforcing_time": 1,
        "daily_cost": 0,
        "color": 'white'
    }
    
    if tiers.tier_type == 'constant':
        originInt["candidate_thresholds"] = [-1]  # Means that there is no lower bound
    elif tiers.tier_type == 'step':
        originInt["candidate_thresholds"] = [[-1], [-0.5]]
    
    if not (originInt in tiers.tier):
        tiers.tier.insert(0, originInt)
    
    given_threshold = eval(args.gt) if args.gt is not None else None
    given_date = eval('dt.datetime({})'.format(args.gd)) if args.gd is not None else None
    
    # if a threshold/threshold+stepping date is given, then it carries out a specific task
    # if not, then search for a policy
    selected_policy = None
    if tiers.tier_type == 'constant':
        if given_threshold is not None:
            selected_policy = MTP.constant_policy(instance, tiers.tier, given_threshold)
    elif tiers.tier_type == 'step':
        if (given_threshold is not None) and (given_date is not None):
            selected_policy = MTP.step_policy(instance, tiers.tier, given_threshold, given_date)
    
    task_str = str(selected_policy) if selected_policy is not None else f'opt{len(tiers.tier)}'
    instance_name = f'{args.f_config[:-5]}_{args.t[:-5]}_{task_str}_rl{str(args.rl)}'
    
    logger.info('============================================================')
    logger.info(f'Instance name: {instance_name}')
    
    # read in the policy upper bound
    if args.pub is not None:
        policy_ub = eval(args.pub)
    else:
        policy_ub = None
    
    
    best_policy_replicas, best_policy, out_file = policy_search_filter(instance=instance,
                                                                        tiers=tiers.tier,
                                                                        obj_func=multi_tier_objective,
                                                                        n_replicas_train=n_replicas_train,
                                                                        n_replicas_test=n_replicas_test,
                                                                        instance_name=instance_name,
                                                                        policy_class=tiers.tier_type,
                                                                        policy=selected_policy,
                                                                        mp_pool=mp_pool,
                                                                        crn_seeds=train_seeds,
                                                                        unique_seeds_ori=test_seeds,
                                                                        forcedOut_tiers=eval(args.fo),
                                                                        redLimit=args.rl,
                                                                        after_tiers=eval(args.aftert),
                                                                        policy_field=args.field,
                                                                        policy_ub=policy_ub)  
        
    if args.plot:
        from pipelinemultitier import multi_tier_pipeline
        multi_tier_pipeline(out_file, instance_name)

#runfile('/Users/ozgesurer/Desktop/Github_Folders/COVID_Staged_Alert/InterventionsMIP/main_downsampling.py', 
#'austin -f setup_data_Final.json -t tiers5_ds_Final.json -train_reps 5 -test_reps 20 -f_config austin_test_IHT_ds.json -n_proc 1 -tr transmission_Final.csv -hos austin_real_hosp_updated.csv')


#runfile('/Users/ozgesurer/Desktop/Github_Folders/COVID_Staged_Alert/InterventionsMIP/main_downsampling.py', 
#'houston -f setup_data_Final.json -t tiers5_ds_Final.json -train_reps 5 -test_reps 20 -f_config houston_test_IHT_ds.json -n_proc 1 -tr transmission_Final_11_1.csv -hos houston_real_hosp_updated.csv')